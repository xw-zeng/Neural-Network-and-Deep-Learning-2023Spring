import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

mpl.use('Agg')

# ## Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 0
gap = 30
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
devices = torch.device("cuda:{}".format(0)
                       if torch.cuda.is_available() else "cpu")
print(devices)
devices_name = torch.cuda.get_device_name()
print(torch.cuda.get_device_name())

# Initialize your data loader and make sure that dataloader works
train_loader = get_cifar_loader(batch_size=128, train=True)
val_loader = get_cifar_loader(batch_size=128, train=False)
for X, y in train_loader:
    try:
        img = np.transpose(X[0], [1, 2, 0])
    except Exception as re:
        print("Error: Please check the data loader.\n")
        print(re)
    else:
        print("The data has been successfully loaded.")
    break


def get_accuracy(model, data_loader, device):
    model.eval()
    correct = num = 0
    with torch.no_grad():
        for _, pack in enumerate(data_loader):
            data, target = pack[0].to(device), pack[1].to(device)
            logit = model(data)
            _, pred = logit.max(1)
            correct += pred.eq(target).sum().item()
            num += data.shape[0]
    # torch.cuda.empty_cache()
    model.train()
    return correct / num


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        from torch.backends import cudnn
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        cudnn.deterministic = True
        cudnn.benchmark = False


def train(model, optimizer, criterion, scheduler=None, epochs_n=20):

    model.to(devices)
    train_accuracy_curve = []
    val_accuracy_curve = []
    iteration = 0
    loss_val = 0
    losses_list = []
    gradients = []
    beta_list = []
    past_grad = None
    past_parm = None

    for _ in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        for data in train_loader:
            iteration += 1
            inputs, labels = data[0].to(devices), data[1].to(devices)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()

            if iteration % gap == 0:
                grad_now = model.classifier[-1].weight.grad.detach().clone()
                parm_now = model.classifier[-1].weight.detach().clone()
                if past_grad is not None:
                    grad_dist = torch.dist(grad_now, past_grad).item()
                    gradients.append(float(grad_dist))
                if past_parm is not None:
                    parm_dist = torch.dist(parm_now, past_parm).item()
                    beta_list.append(float(grad_dist / (parm_dist + 1e-3)))
                past_grad = grad_now
                past_parm = parm_now
                losses_list.append(loss_val / gap)
                loss_val = 0

    return tuple(losses_list), gradients, beta_list, train_accuracy_curve, val_accuracy_curve


def plot_loss_landscape(ax_plt, list_BA: np.ndarray, list_BN: np.ndarray):
    downwards_BA = np.min(list_BA, axis=0)
    upwards_BA = np.max(list_BA, axis=0)
    i = np.arange(0, downwards_BA.__len__(), 1, dtype=int) * gap
    upwards_BA = upwards_BA
    downwards_BA = downwards_BA
    ax_plt.plot(i, downwards_BA, color="g", alpha=0.8)
    ax_plt.plot(i, upwards_BA, color="g", alpha=0.8)
    ax_plt.get_lines()[0].set_linewidth(0.8)
    ax_plt.get_lines()[1].set_linewidth(0.8)
    ax_plt.fill_between(i, downwards_BA, upwards_BA,
                        where=(upwards_BA > downwards_BA),
                        color="g", alpha=0.4, label="Standard VGG")

    downwards_BN = np.min(list_BN, axis=0)
    upwards_BN = np.max(list_BN, axis=0)
    i = np.arange(0, downwards_BN.__len__(), 1, dtype=int) * gap
    upwards_BN = upwards_BN
    downwards_BN = downwards_BN
    ax_plt.plot(i, downwards_BN, color="firebrick", alpha=0.8)
    ax_plt.plot(i, upwards_BN, color="firebrick", alpha=0.8)
    ax_plt.get_lines()[2].set_linewidth(0.8)
    ax_plt.get_lines()[3].set_linewidth(0.8)
    ax_plt.fill_between(i, downwards_BN, upwards_BN,
                        where=(upwards_BN > downwards_BN),
                        color="firebrick", alpha=0.4, label="Standard VGG + BachNorm")

    ax_plt.legend()
    ax_plt.set(title='Loss Landscape', ylabel='Loss Landscape', xlabel='Steps')


def plot_gradient_dist(ax_plt, list_BA: np.ndarray, list_BN: np.ndarray):
    downwards_BA = np.min(list_BA, axis=0)
    upwards_BA = np.max(list_BA, axis=0)
    i = np.arange(0, downwards_BA.__len__(), 1, dtype=int) * gap
    upwards_BA = upwards_BA
    downwards_BA = downwards_BA
    ax_plt.plot(i, downwards_BA, color="g", alpha=0.8)
    ax_plt.plot(i, upwards_BA, color="g", alpha=0.8)
    ax_plt.get_lines()[0].set_linewidth(0.8)
    ax_plt.get_lines()[1].set_linewidth(0.8)
    ax_plt.fill_between(i, downwards_BA, upwards_BA,
                        where=(upwards_BA > downwards_BA),
                        color="g", alpha=0.4, label="Standard VGG")

    downwards_BN = np.min(list_BN, axis=0)
    upwards_BN = np.max(list_BN, axis=0)
    i = np.arange(0, downwards_BN.__len__(), 1, dtype=int) * gap
    upwards_BN = upwards_BN
    downwards_BN = downwards_BN
    ax_plt.plot(i, downwards_BN, color="firebrick", alpha=0.8)
    ax_plt.plot(i, upwards_BN, color="firebrick", alpha=0.8)
    ax_plt.get_lines()[2].set_linewidth(0.8)
    ax_plt.get_lines()[3].set_linewidth(0.8)
    ax_plt.fill_between(i, downwards_BN, upwards_BN,
                        where=(upwards_BN > downwards_BN),
                        color="firebrick", alpha=0.4, label="Standard VGG + BachNorm")
    ax_plt.legend()
    ax_plt.set(title='Gradient Distance', ylabel='gradient-dist', xlabel='Steps')


def plot_beta_smooth(ax_plt, list_BA: np.ndarray, list_BN: np.ndarray):
    beta_BN = np.max(np.asarray(list_BN), axis=0)
    beta_BA = np.max(np.asarray(list_BA), axis=0)
    Step = np.arange(0, beta_BA.__len__(), 1, dtype=int) * gap
    ax_plt.plot(Step, beta_BA, color="g", label="Standard VGG")
    ax_plt.plot(Step, beta_BN, color="firebrick", label="Standard VGG + BachNorm")
    ax_plt.get_lines()[0].set_linewidth(0.8)
    ax_plt.get_lines()[1].set_linewidth(0.8)
    ax_plt.legend()
    ax_plt.set(title='Beta-Smoothness', ylabel='beta-smooth', xlabel='Steps')


if __name__ == '__main__':
    epo = 20
    # learning_list = [0.05, 0.075, 0.1, 0.125, 0.15]
    learning_list = [1e-3, 2e-3, 1e-4, 5e-4]
    grad_list_BA = []
    loss_list_BA = []
    beta_list_BA = []
    set_random_seeds(seed_value=2023, device=devices_name)

    for learning_rate in learning_list:
        print('======== Start Training ========')
        print(f'Learning rate: {learning_rate}')
        print('======== Standard VGG-A ========')
        model_Vgg = VGG_A()
        Opt = torch.optim.SGD(model_Vgg.parameters(), lr=learning_rate)
        Loss = nn.CrossEntropyLoss()
        loss_l, grads, beta, _, _ = train(model_Vgg, Opt, Loss, epochs_n=epo)
        grad_list_BA.append(grads)
        loss_list_BA.append(loss_l)
        beta_list_BA.append(beta)

    grad_list_BN = []
    loss_list_BN = []
    beta_list_BN = []
    for learning_rate in learning_list:
        print('======== Start Training ========')
        print(f'Learning rate: {learning_rate}')
        print('======== VGG-A + BatchNorm ========')
        model_BN_Vgg = VGG_A_BatchNorm()
        Opt = torch.optim.SGD(model_BN_Vgg.parameters(), lr=learning_rate)
        Loss = nn.CrossEntropyLoss()
        loss_l, grads, beta, _, _ = train(model_BN_Vgg, Opt, Loss, epochs_n=epo)
        grad_list_BN.append(grads)
        loss_list_BN.append(loss_l)
        beta_list_BN.append(beta)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(27, 6), dpi=800)
    ax1 = fig.add_subplot(131)
    plot_loss_landscape(ax1, np.asarray(loss_list_BA), np.asarray(loss_list_BN))
    ax2 = fig.add_subplot(132)
    plot_gradient_dist(ax2, np.asarray(grad_list_BA), np.asarray(grad_list_BN))
    ax3 = fig.add_subplot(133)
    plot_beta_smooth(ax3, np.asarray(beta_list_BA), np.asarray(beta_list_BN))
    plt.savefig('./results/BatchNorm_res.png')
