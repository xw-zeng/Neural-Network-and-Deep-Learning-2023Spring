import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import copy
import os
import argparse
import logging
import models
from utils import (
    Logger,
    count_parameters,
    update_nesterov,
    patch_whitening,
    label_smoothing_loss,
    update_ema,
    load_cifar10,
    random_crop,
)

parser = argparse.ArgumentParser(description="CIFAR-10 Classification")
parser.add_argument('--model', default='FastResNet9', type=str, help="FastResNet9")
parser.add_argument('--batch-size', default=512, type=int, help='Batch size')
parser.add_argument('--max-epoch', default=20, type=int, help='Max training epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight-decay', default=0.256, type=float, help='Weight decay lambda')
parser.add_argument('--weight-decay-bias', default=0.004, type=float, help='Weight decay bias')
parser.add_argument('--ema-update-freq', default=5, type=int, help='Ema Update Frequency')
parser.add_argument('--ema-rho', default=0.99 ** 5, type=int, help='Ema Update Rho')
args = parser.parse_args()

lr_schedule = torch.cat([
    torch.linspace(0, 2e-3, 194),
    torch.linspace(2e-3, 2e-4, 582),
])
lr_schedule_bias = 64.0 * lr_schedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16 if device.type != 'cpu' else torch.float32

if device.type == 'cuda':
    cudnn.benchmark = True

if not os.path.isdir('logs'):
    os.mkdir('logs')
logger = Logger(
    log_file_name=f'./logs/{args.model.lower()}.txt',
    log_level=logging.DEBUG,
    logger_name=args.model.lower(),
).get_log()

best_acc = 0  # Best test accuracy
start_epoch = 0  # Start from epoch 0 or last checkpoint epoch

torch.manual_seed(0)

# Data
logger.info(' ======== Data Preprocessing ======== ')
train_data, train_targets, test_data, test_targets = load_cifar10(device, dtype)

# Model
logger.info(f' ======== Model: {args.model} ======== ')
weights = patch_whitening(train_data[:10000, :, 4:-4, 4:-4])  # Compute special weights for first layer
net = models.FastResNet9(weights, c_in=3, c_out=10, scale_out=0.125)
logger.info(f' Total Parameters: {count_parameters(net)}')
net.to(dtype)  # Convert model weights to half precision

# Convert BatchNorm back to single precision for better accuracy
for module in net.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.float()

net.to(device)  # Upload model to GPU

# Collect weights and biases and create nesterov velocity values
weights = [
    (w, torch.zeros_like(w))
    for w in net.parameters()
    if w.requires_grad and len(w.shape) > 1
]
biases = [
    (w, torch.zeros_like(w))
    for w in net.parameters()
    if w.requires_grad and len(w.shape) <= 1
]

net_test = copy.deepcopy(net)
criterion = label_smoothing_loss
batch_count = 0

# Training
def train():
    global batch_count
    tr_loss = 0
    correct = 0
    total = 0

    # Randomly shuffle training data
    indices = torch.randperm(len(train_data), device=device)
    data = train_data[indices]
    targets = train_targets[indices]

    # Crop random 32x32 patches from 40x40 training data
    data = [
        random_crop(data[i: i + args.batch_size], crop_size=(32, 32))
        for i in range(0, len(data), args.batch_size)
    ]
    data = torch.cat(data)

    # Randomly flip half the training data
    data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])

    for idx in range(0, len(data), args.batch_size):
        # discard partial batches
        if idx + args.batch_size > len(data):
            break

        # Slice batch from data
        inputs = data[idx: idx + args.batch_size]
        target = targets[idx: idx + args.batch_size]
        batch_count += 1

        # Compute new gradients
        net.zero_grad()
        net.train(True)

        y_hat = net(inputs)
        loss = criterion(y_hat, target)
        loss.sum().backward()

        lr_index = min(batch_count, len(lr_schedule) - 1)
        lr = lr_schedule[lr_index]
        lr_bias = lr_schedule_bias[lr_index]

        # Update weights and biases of training model
        update_nesterov(weights, lr, args.weight_decay, args.momentum)
        update_nesterov(biases, lr_bias, args.weight_decay_bias, args.momentum)

        # Update validation model with exponential moving averages
        if (idx // args.batch_size % args.ema_update_freq) == 0:
            update_ema(net, net_test, args.ema_rho)

        tr_loss += loss.sum()
        _, predicted = y_hat.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = tr_loss / (idx * args.batch_size)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def test(epoch):
    global best_acc
    ts_loss = 0
    correct = 0
    total = 0

    for idx in range(0, len(test_data), args.batch_size):
        net_test.train(False)

        # Test data augmentation: Test model on raw and flipped data
        raw_inputs = test_data[idx: idx + args.batch_size]
        flipped_inputs = torch.flip(raw_inputs, [-1])
        target = test_targets[idx: idx + args.batch_size]

        y_hat1 = net_test(raw_inputs).detach()
        y_hat2 = net_test(flipped_inputs).detach()
        y_hat = torch.mean(torch.stack([y_hat1, y_hat2], dim=0), dim=0)

        loss = criterion(y_hat, target)
        ts_loss += loss.sum()
        _, predicted = y_hat.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = ts_loss / ((idx + 1) * args.batch_size)
    epoch_acc = correct / total

    # Save checkpoint
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        state = {
            'net': net_test.state_dict(),
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

logger.info(' ======== Start Training ======== ')

for epoch in range(start_epoch, start_epoch + args.max_epoch):
    start = time.time()
    train_loss, train_acc = train()
    end = time.time()
    logger.info(f' ======== Epoch: [{epoch + 1}/{start_epoch + args.max_epoch}] ========')
    logger.info(
        f" Train Loss: {float(train_loss):.4f} | Train Acc: {train_acc * 100:.2f}% | Cost Time: {end - start:.4f}s"
    )

    test_loss, test_acc = test(epoch)
    logger.info(
        f" Test Loss: {float(test_loss):.4f} | Test Acc: {test_acc * 100:.2f}% | Best Acc: {best_acc * 100:.2f}%"
    )

    train_losses.append(float(train_loss))
    train_accuracies.append(train_acc)
    test_losses.append(float(test_loss))
    test_accuracies.append(test_acc)

logger.info(' ======== Training Finished ======== ')

# Plot
plt.style.use('ggplot')
fig = plt.figure(figsize=(15, 5), dpi=800)

ax1 = fig.add_subplot(121)
ax1.plot(train_losses, label='train')
ax1.plot(test_losses, label='test')
ax1.legend()
ax1.set(title='Loss Curve', ylabel='Loss', xlabel='Epoch')

ax2 = fig.add_subplot(122)
ax2.plot(train_accuracies, label='train')
ax2.plot(test_accuracies, label='test')
ax2.legend()
ax2.set(title='Accuracy Curve', ylabel='Accuracy', xlabel='Epoch')

if not os.path.isdir('results'):
    os.mkdir('results')
plt.savefig(f'./results/{args.model.lower()}.png')

Steps = np.arange(len(train_accuracies))
res = pd.DataFrame(Steps)
res['train_loss'] = train_losses
res['test_loss'] = test_losses
res['train_acc'] = train_accuracies
res['test_acc'] = test_accuracies
res.to_csv(f'./results/{args.model.lower()}.csv', index=False)
