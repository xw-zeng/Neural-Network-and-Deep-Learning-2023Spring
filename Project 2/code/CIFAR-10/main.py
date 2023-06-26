import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
import argparse
import logging
import models
from utils import (
    Logger,
    count_parameters
)

parser = argparse.ArgumentParser(description="CIFAR-10 Classification")
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=0.1, type=float, help='Initial Learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay lambda')
parser.add_argument('--max-epoch', default=20, type=int, help='Max training epochs')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument(
    '--model',
    default='ResNet18',
    type=str,
    help='LeNet, AlexNet, ResNet18, ResNet34, ResNet50, ResNeXt29_2x32d, ResNeXt29_2x64d, ResNeXt29_32x4d, \
    DenseNet100bc, DenseNet121, WideResNet16x8, WideResNet28x10, GeResNeXt29_8x64d',
)
parser.add_argument(
    '--optim', 
    default='SGD', 
    type=str, 
    help='SGD, Adagrad, Adadelta, Adam, Adamax'
)
parser.add_argument(
    '--lr-rule',
    default='Plateau',
    type=str,
    help='Plateau, CosAnneal, Step, Linear, Exp')
parser.add_argument(
    '--loss', 
    default='CE', 
    type=str, 
    help='CE, BCElogit, MSE, Huber, Soft'
)
args = parser.parse_args()

if not os.path.isdir('logs'):
    os.mkdir('logs')
logger = Logger(
    log_file_name=f'./logs/{args.model.lower()}.txt',
    log_level=logging.DEBUG,
    logger_name=args.model.lower(),
).get_log()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # Best test accuracy
start_epoch = 0  # Start from epoch 0 or last checkpoint epoch

# Data
logger.info(' ======== Data Preprocessing ======== ')

transform_train = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
train_batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=train_batch_size, shuffle=True,
    # num_workers=10, pin_memory=True
)

transform_test = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]
)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False,
    # num_workers=10, pin_memory=True
)

# Model
logger.info(f' ======== Model: {args.model} ======== ')
net = getattr(models, args.model)().to(device)
logger.info(f' Total Parameters: {count_parameters(net)}')

if device == 'cuda':
    net = nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    logger.info(f' ======== Resume from checkpoint ======== ')
    assert os.path.isdir('checkpoints'), "Error: no checkpoints directory found!"
    # checkpoint = torch.load(f'./checkpoints/checkpoint_{args.model}.pth')
    checkpoint = torch.load(f'./checkpoints/resnet18.pth')
    net.load_state_dict(checkpoint)
    # net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'BCElogit':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss == 'MSE':
    criterion = nn.MSELoss()
elif args.loss == 'Huber':
    criterion = nn.HuberLoss()
elif args.loss == 'Soft':
    criterion = nn.MultiLabelSoftMarginLoss()

if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
    momentum=0.9, weight_decay=args.weight_decay)
elif args.optim == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr,
    weight_decay=args.weight_decay)
elif args.optim == 'Adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr,
    weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
    weight_decay=args.weight_decay)
elif args.optim == 'Adamax':
    optimizer = optim.Adamax(net.parameters(), lr=args.lr,
    weight_decay=args.weight_decay)

if args.lr_rule == 'Plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6
    )

# Training
def train(l):
    tr_loss = 0
    correct = 0
    total = 0
    net.train()

    for idx, (X, labels) in enumerate(train_loader):
        X, labels = X.to(device), labels.to(device)
        labels_onehot = F.one_hot(labels, num_classes=10).to(torch.float)
        optimizer.zero_grad()
        y_hat = net(X)
        if l == 'CE':
            loss = criterion(y_hat, labels)
        else:
            loss = criterion(y_hat, labels_onehot)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        _, predicted = y_hat.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = tr_loss / ((idx + 1) * train_batch_size)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def test(epoch, l):
    global best_acc
    ts_loss = 0
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for idx, (X, labels) in enumerate(test_loader):
            X, labels = X.to(device), labels.to(device)
            labels_onehot = F.one_hot(labels, num_classes=10).to(torch.float)
            y_hat = net(X)
            if l == 'CE':
                loss = criterion(y_hat, labels)
            else:
                loss = criterion(y_hat, labels_onehot)
            ts_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = ts_loss / ((idx + 1) * test_batch_size)
    epoch_acc = correct / total

    # Save checkpoint
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        state = {
            'net': net.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/checkpoint_{args.model.lower()}.pth')

    return epoch_loss, epoch_acc


# Training
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
lr_schedule = []

logger.info(' ======== Start Training ======== ')

for epoch in range(start_epoch, start_epoch + args.max_epoch):
    start = time.time()
    train_loss, train_acc = train(args.loss)
    end = time.time()
    logger.info(f' ======== Epoch: [{epoch + 1}/{start_epoch + args.max_epoch}] ========')
    logger.info(f" Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    logger.info(
        f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Cost Time: {end - start:.4f}s"
    )

    test_loss, test_acc = test(epoch, args.loss)
    logger.info(
        f" Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}% | Best Acc: {best_acc * 100:.2f}%"
    )

    if args.lr_rule == 'Plateau':
        scheduler.step(test_loss)
    elif args.lr_rule == 'Step':
        if (epoch + 1) % 5 == 0:
            optimizer.param_groups[0]["lr"] *= 0.5
    elif args.lr_rule == 'Linear':
        if epoch < 15:
            optimizer.param_groups[0]["lr"] -= 0.00066
    elif args.lr_rule == 'CosAnneal':
        optimizer.param_groups[0]["lr"] = 1e-4 + 0.5 * (1e-2 - 1e-4) * (
                1 + np.cos(epoch / (start_epoch + args.max_epoch) * np.pi))
    elif args.lr_rule == 'Exp':
        optimizer.param_groups[0]["lr"] = 0.01 * 0.9 ** epoch

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    lr_schedule.append(optimizer.param_groups[0]['lr'])

logger.info(' ======== Training Finished ======== ')

# Plot
plt.style.use('ggplot')
fig = plt.figure(figsize=(22.5, 5), dpi=800)

ax1 = fig.add_subplot(131)
ax1.plot(train_losses, label='train')
ax1.plot(test_losses, label='test')
ax1.legend()
ax1.set(title='Loss Curve', ylabel='Loss', xlabel='Epoch')

ax2 = fig.add_subplot(132)
ax2.plot(train_accuracies, label='train')
ax2.plot(test_accuracies, label='test')
ax2.legend()
ax2.set(title='Accuracy Curve', ylabel='Accuracy', xlabel='Epoch')

ax3 = fig.add_subplot(133)
ax3.plot(lr_schedule)
ax3.set(title='Learning Rate Curve', ylabel='Learning Rate', xlabel='Epoch')

if not os.path.isdir('results'):
    os.mkdir('results')
plt.savefig(f'./results/{args.model.lower()}.png')

Steps = np.arange(len(train_accuracies))
res = pd.DataFrame(Steps)
res['train_loss'] = train_losses
res['test_loss'] = test_losses
res['train_acc'] = train_accuracies
res['test_acc'] = test_accuracies
res['lr'] = lr_schedule
res.to_csv(f'./results/{args.model.lower()}.csv', index=False)
