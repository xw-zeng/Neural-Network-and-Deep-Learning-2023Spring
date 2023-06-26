import os
import sys
sys.path.append('E:/复旦/大三下/0神经网络与深度学习/Project/Finalpj')
from DessiLBI.ADAM_code.slbi_toolbox import SLBI_ToolBox_ADAM
from DessiLBI.code.slbi_toolbox import SLBI_ToolBox_Base
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from data_loader import load_data
from torch.backends import cudnn
import argparse
import lenet
import matplotlib.pyplot as plt
import numpy as np


cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--interval", default=20, type=int)
parser.add_argument("--kappa", default=1, type=int)
parser.add_argument("--dataset", default='MNIST', type=str)
parser.add_argument("--train", default=True, type=str2bool)
parser.add_argument("--download", default=True, type=str2bool)
parser.add_argument("--shuffle", default=True, type=str2bool)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=False, type=str2bool)
parser.add_argument("--epoch", default=32, type=int)
parser.add_argument("--model_name", default='lenet', type=str)
parser.add_argument("--gpu_num", default='0', type=str)
parser.add_argument("--mu", default=40, type=int)
parser.add_argument("--gap", default=20, type=int)
args = parser.parse_args()
name_list = []
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
torch.cuda.empty_cache()

train_loader = load_data(dataset=args.dataset, train=True, download=args.download, batch_size=args.batch_size,
                         shuffle=args.shuffle)
test_loader = load_data(dataset=args.dataset, train=False, download=False, batch_size=64, shuffle=False)


def model_init():
    model = lenet.Net().to(device)

    if args.parallel:
        model = nn.DataParallel(model)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    layer_lists = []
    for name, p in model.named_parameters():
        name_list.append(name)
        print(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_lists.append(name)

    return model, layer_lists


def train(model, optimizer):
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
                pa = optimizer.calculate_path('conv2.weight')
                path.append(pa.to('cpu').numpy())
                correct = num = 0
                loss_val = 0
        print('*******************************')
        print('Test Model')
        evaluate_batch(model, test_loader, device)
    # print('*******************************')
    return Loss, acc, sparsity, path


if __name__ == '__main__':

    sys.stdout = open('./log_train.txt', 'w')

    net1, layer_list = model_init()
    optimizer_adam = SLBI_ToolBox_ADAM(net1.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu)
    optimizer_adam.assign_name(name_list)
    optimizer_adam.initialize_slbi(layer_list)
    ADAM_LOSS, ADAM_acc, ADAM_SPA, ADAM_PATH = train(net1, optimizer_adam)
    save_model_and_optimizer(net1, optimizer_adam, 'lenet_adam.pth')

    net2, layer_list = model_init()
    args.lr = 0.1
    optimizer_base = SLBI_ToolBox_Base(net2.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu)
    optimizer_base.assign_name(name_list)
    optimizer_base.initialize_slbi(layer_list)
    Base_Loss, Base_acc, Base_SPA, Base_PATH = train(net2, optimizer_base)
    save_model_and_optimizer(net2, optimizer_base, 'lenet_base.pth')

    fig = plt.figure(figsize=(36, 6), dpi=800)
    ax1 = fig.add_subplot(141)
    iterations = np.arange(1, len(ADAM_acc) + 1, dtype=int) * args.gap
    ax1.plot(iterations[1:60], np.asarray(ADAM_LOSS)[1:60], color="firebrick", label='ADAM')
    ax1.plot(iterations[1:60], np.asarray(Base_Loss)[1:60], color="g", label='BASE')
    ax1.legend()
    ax1.set(title='Loss',
            ylabel='Loss',
            xlabel='Steps')

    ax2 = fig.add_subplot(142)
    ax2.plot(iterations[1:60], np.asarray(ADAM_acc)[1:60], color="firebrick", label='ADAM')
    ax2.plot(iterations[1:60], np.asarray(Base_acc)[1:60], color="g", label='BASE')
    ax2.legend()
    ax2.set(title='Accuracy',
            ylabel='Accuracy',
            xlabel='Steps')

    ax3 = fig.add_subplot(143)
    spaa = np.asarray(ADAM_SPA)
    spab = np.asarray(Base_SPA)
    for i in range(5):
        label = 'ADAM: ' + layer_list[i].split('.')[0]
        ax3.plot(iterations, spaa[:, i], label=label)
    for i in range(5):
        label = 'BASE: ' + layer_list[i].split('.')[0]
        ax3.plot(iterations, spab[:, i], label=label)

    ax3.legend(loc='center right')
    ax3.set(title='Sparsity',
            ylabel='Percentage',
            xlabel='Steps')

    ax4 = fig.add_subplot(144)
    ax4.plot(iterations, ADAM_PATH, color="firebrick")
    ax4.plot(iterations, Base_PATH, color='g')
    ax4.plot([], [], color="firebrick", label='ADAM')
    ax4.plot([], [], color='g', label='BASE')
    ax4.legend()
    ax4.set(title='Gamma Path',
            ylabel='Gamma',
            xlabel='Steps')

    plt.savefig('./results/res.png')
