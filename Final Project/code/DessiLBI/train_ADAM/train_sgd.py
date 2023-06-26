import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import load_data
from lenet import Net
import numpy as np

# loading
train_loader = load_data(dataset='MNIST', train=True, download=True, batch_size=128, shuffle=True)
test_loader = load_data(dataset='MNIST', train=False, download=False, batch_size=64, shuffle=False)
device = torch.device('cuda')

def train(lr, interval, epoch=2):
    train_accs  = []
    test_accs = []
    for ep in range(epoch):
        lr = lr * (0.1 ** (ep //interval))
        loss_val = 0
        correct = 0
        num = 0
        model.train()
        for iter, pack in enumerate(train_loader):
            data, target = pack[0].to(device), pack[1].to(device)
            logits = model(data)
            loss = F.nll_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = logits.max(1)
            loss_val += loss.item()
            num += data.shape[0]
        loss_val /= num 

        train_acc = get_accuracy(train_loader)
        test_acc = get_accuracy(test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print('epoch', ep , 'loss', loss_val, 'acc', test_acc)
        correct = num = 0
        loss_val = 0

def get_accuracy(test_loader):
    model.eval()
    correct = 0
    num = 0
    for _, pack in enumerate(test_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
    acc = correct / num 
    return acc 

def plot_weight_conv3(weight, title, savename):
    plt.figure(figsize=(5, 5), dpi=500)
    plt.clf()
    plt.imshow(weight, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig('./results/' + savename + '.png')

if __name__=='__main__':
    lr = 1e-2
    kappa = 1
    mu = 40
    interval = 20
    epoch = 32

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train(lr, interval, epoch=epoch)
    torch.save(model.state_dict(), 'lenet_sgd.pth')
    # visulization
    weight = model.conv3.weight.clone().detach().cpu().numpy()
    H=10; W=10
    weights = np.zeros((H*5,W*5))
    for i in range(H):
        for j in range(W):
            weights[i*5:i*5+5, j*5:j*5+5] = weight[i][j]
    weights = np.abs(weights)
    plot_weight_conv3(weights, title='BASE: conv3 weight', savename='conv3_weight_base')

