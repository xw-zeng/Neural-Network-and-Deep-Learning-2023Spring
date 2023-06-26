import os
import sys
sys.path.append('E:/复旦/大三下/0神经网络与深度学习/Project/Finalpj')
from DessiLBI.ADAM_code.slbi_toolbox import SLBI_ToolBox_ADAM
from utils_adam import *
from torch.backends import cudnn
import argparse
from models.vgg import VGG_A_BatchNorm
from loaders import *
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--interval", default=20, type=int)
parser.add_argument("--kappa", default=1, type=int)
parser.add_argument("--dataset", default='CIFAR10', type=str)
parser.add_argument("--train", default=True, type=str2bool)
parser.add_argument("--download", default=True, type=str2bool)
parser.add_argument("--shuffle", default=True, type=str2bool)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=False, type=str2bool)
parser.add_argument("--epoch", default=32, type=int)
parser.add_argument("--model_name", default='VGG', type=str)
parser.add_argument("--gpu_num", default='0', type=str)
parser.add_argument("--mu", default=20, type=int)
parser.add_argument("--gap", default=20, type=int)
args = parser.parse_args()
name_list = []
torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(0)
                      if args.use_cuda and torch.cuda.is_available() else "cpu")
print(device)
devices_name = torch.cuda.get_device_name()
print(torch.cuda.get_device_name())

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)


if __name__ == '__main__':

    sys.stdout = open('./log_vgg.txt', 'w')

    model = VGG_A_BatchNorm().to(device)
    layer_list = []
    for name, p in model.named_parameters():
        name_list.append(name)
        print(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_list.append(name)

    # Initialize the optimizer
    optimizer = SLBI_ToolBox_ADAM(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu)
    optimizer.assign_name(name_list)
    optimizer.initialize_slbi(layer_list)

    sparsity = []
    Loss = []
    acc = []
    path = []
    all_num = args.epoch * len(train_loader)
    iteration = 0
    print('num of all step:', all_num)
    print('num of step per epoch:', len(train_loader))
    for ep in range(args.epoch):
        model.train()
        descent_lr(args.lr, ep, optimizer, args.interval)
        loss_val = 0
        correct = num = 0
        for ITER, pack in enumerate(train_loader):
            iteration += 1
            data, target = pack[0].to(device), pack[1].to(device)
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = logits.max(1)
            loss_val += loss.item()
            correct += pred.eq(target).sum().item()
            num += data.shape[0]
            if iteration % args.gap == 0:
                print('*******************************')
                print('epoch : ', ep + 1)
                print(f'iteration:{iteration} \t | Train ACC: {correct / num}')
                print('loss : ', loss_val / args.gap)
                Loss.append(loss_val / args.gap)
                acc.append(correct / num)
                SPA = optimizer.calculate_sparsity()
                sparsity.append(SPA)
                pa = optimizer.calculate_path('classifier.3.weight')
                path.append(pa.cpu().numpy())
                correct = num = 0
                loss_val = 0
        optimizer.calculate_all_w_star(ep)
        print('*******************************')
        print('Test Model')
        evaluate_batch(model, val_loader, device)

    fig = plt.figure(figsize=(22.5, 5), dpi=800)
    ax1 = fig.add_subplot(131)
    iterations = np.arange(1, len(acc) + 1, dtype=int) * args.gap
    ax1.plot(iterations, np.asarray(acc), color="firebrick", label='ADAM')
    ax1.legend()
    ax1.set(title='Accuracy',
            ylabel='Accuracy',
            xlabel='Steps')
    
    ax2 = fig.add_subplot(132)
    ax2.plot(iterations, np.asarray(Loss), color="firebrick", label='ADAM')
    ax2.legend()
    ax2.set(title='Loss',
            ylabel='Loss',
            xlabel='Steps')

    ax3 = fig.add_subplot(133)
    spa = np.asarray(sparsity)
    _, fil = spa.shape
    for i in range(fil):
        label = 'ADAM: ' + layer_list[i].replace('.weight', '')
        if (i == 0) or (spa[:, i] != spa[:, i - 1]).all():
                ax3.plot(iterations, spa[:, i], label=label)

    ax3.legend()
    ax3.set(title='Sparsity',
            ylabel='Percentage',
            xlabel='Steps')

    plt.savefig('result.png')
