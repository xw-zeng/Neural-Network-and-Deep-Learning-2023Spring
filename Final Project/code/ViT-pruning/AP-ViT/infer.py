# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from collections import OrderedDict

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def top5_accuracy(preds, labels):
    return ((preds - labels) == 0).any(axis=1).mean()


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def infer(args, model, clip_layer_id, stream_id, infer_mode, threshold):
    eval_losses = AverageMeter()

    train_loader, test_loader = get_loader(args)

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label, top5_preds = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    model.to(args.device)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x, stream_id=stream_id, clip_layer_id=clip_layer_id, infer_mode=infer_mode, threshold=threshold)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)
            _, preds_5 = torch.topk(logits, k=5)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            top5_preds.append(preds_5.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            top5_preds[0] = np.append(
                top5_preds[0], preds_5.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, top5_preds = all_preds[0], all_label[0], top5_preds[0]
    accuracy = simple_accuracy(all_preds, all_label)
    top5_acc = top5_accuracy(top5_preds, all_label[:, np.newaxis])
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Top 1 Accuracy: %2.5f" % accuracy)
    logger.info("Valid Top 5 Accuracy: %2.5f" % top5_acc)

    return accuracy, top5_acc


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="output/1_cifar10_checkpoint.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=72, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=500, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=30000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--pruned_ratio", default=0.1, type=float,
                        help="Pruning ratio.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=3,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s" %
                   (args.local_rank, args.device, args.n_gpu))

    # Set clip_layer_id
    # clip_layer_id = [2, 5, 8]
    clip_layer_id = list(range(12))

    # Model & Tokenizer Setup
    config = CONFIGS[args.model_type]
    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                              clip_layer_id=clip_layer_id)
    assert args.eval_batch_size == 1

    # checkpoint = torch.load(args.pretrained_dir)
    # check = OrderedDict((k.replace('module.',''), v) for k, v in checkpoint.items())
    # model.load_state_dict(check)

    # Infer
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "1_%s_checkpoint.pth" % args.name)))
    # 这里的pruned ratio与threshold相对应的字典需要根据训练好的模型自行修改
    # 因为模型只能修改threshold，然后根据threshold算sparsity
    # 这里的字典是我跑的时候适用的
    threshold = {0: 1.0, 0.1: 0.525, 0.2: 0.49, 0.3: 0.467, 0.4: 0.451, 0.5: 0.439,
                 0.6: 0.431, 0.7: 0.419, 0.8: 0.40495, 0.9: 0.385}
    infer(args, model, clip_layer_id, stream_id=1, infer_mode=True, threshold=threshold[args.pruned_ratio])


if __name__ == "__main__":
    main()
