from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import time
import os
import pickle
import torch.backends.cudnn as cudnn
import yaml

import opts_eval
from misc import utils, eval_utils
from misc import AttModel
import yaml
import json

# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def eval(opt):
    model.eval()
    #########################################################################################
    # eval begins here
    #########################################################################################
    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []
    count = 0
    for step in range(len(dataloader_val)):
        data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data

        proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]

        input_imgs.resize_(img.size()).copy_(img)
        input_seqs.resize_(iseq.size()).copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.resize_(num.size()).copy_(num)
        input_ppls.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.resize_(img.size()).copy_(img)

        eval_opt = {'sample_max': 1, 'beam_size': 2, 'inference_mode': True, 'tag_size': opt.cbs_tag_size}
        seq, bn_seq, fg_seq = model(input_imgs, input_seqs, gt_seqs,
                                    input_num, input_ppls, gt_bboxs, mask_bboxs, 'sample', eval_opt)

        sents = utils.decode_sequence(dataset.itow, dataset.itod, dataset.ltow, dataset.itoc, dataset.wtod,
                                      seq.data, bn_seq.data, fg_seq.data, opt.vocab_size, opt)
        for k, sent in enumerate(sents):
            entry = {'image_id': img_id[k].item(), 'caption': sent}
            predictions.append(entry)
            if num_show < 100:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                num_show += 1

        if count % 100 == 0:
            print(count)
        count += 1

    print('Total image to be evaluated %d' % (len(predictions)))
    lang_stats = None
    if opt.language_eval == 1:
        lang_stats = utils.noc_eval(predictions, str(1), 'test', opt)
        print(f"Saving scores into '{opt.checkpoint_path}'")
        with open(os.path.join(opt.checkpoint_path, 'lang_stats.json'), 'w') as f:
            json.dump(lang_stats, f)
        print("Done!")
        print(f"Now saving images and captions into '{opt.checkpoint_path}'")
        with open(os.path.join(opt.checkpoint_path, 'preds.json'), 'w') as f:
            json.dump(predictions, f)
        with open(os.path.join(opt.checkpoint_path, 'sents.json'), 'w') as f:
            json.dump(sents, f)
        print("Done!")

    return lang_stats


if __name__ == '__main__':
    opt = opts_eval.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.safe_load(handle)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    cudnn.benchmark = True

    from misc.dataloader_coco import DataLoader

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset = DataLoader(opt, split='train')
    dataset_val = DataLoader(opt, split='test')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                                 shuffle=False, num_workers=opt.num_workers)

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)

    opt.vocab_size = dataset.vocab_size
    opt.detect_size = dataset.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset.fg_size
    opt.fg_mask = torch.from_numpy(dataset.fg_mask).byte()
    opt.glove_fg = torch.from_numpy(dataset.glove_fg).float()
    opt.glove_clss = torch.from_numpy(dataset.glove_clss).float()
    opt.glove_w = torch.from_numpy(dataset.glove_w).float()
    opt.st2towidx = torch.from_numpy(dataset.st2towidx).long()

    opt.itow = dataset.itow
    opt.itod = dataset.itod
    opt.ltow = dataset.ltow
    opt.itoc = dataset.itoc

    model = AttModel.TopDownModel(opt)

    infos = {}
    histories = {}
    model_path = os.path.join(opt.start_from, 'model-best.pth')
    info_path = os.path.join(opt.start_from, 'infos_-best' + opt.id + '.pkl')

    # open old infos and check if models are compatible
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    print('Loading the model %s...' % model_path)
    model.load_state_dict(torch.load(model_path))

    model._reinit_word_weight(opt, dataset.ctoi, dataset.wtoi)

    model.cuda()

    eval(opt)
