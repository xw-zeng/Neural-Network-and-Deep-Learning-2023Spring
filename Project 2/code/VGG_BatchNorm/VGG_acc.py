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
    torch.cuda.empty_cache()
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


def train(model, optimizer, criterion, train_loaders, val_loaders,
          scheduler=None, epochs_n=100, best_model_path=None):

    model.to(devices)
    train_accuracy_curve = []
    val_accuracy_curve = []
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0
    iteration = 0
    gap = 100

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

            if iteration % gap == 0:
                acc_train = get_accuracy(model, train_loaders, devices)
                acc_val = get_accuracy(model, val_loaders, devices)
                train_accuracy_curve.append(acc_train)
                val_accuracy_curve.append(acc_val)
                max_val_accuracy = max(acc_val, max_val_accuracy)

    print(f'======== Finish Training: best acc: {max_val_accuracy} ========')
    return train_accuracy_curve, val_accuracy_curve


if __name__ == '__main__':
    epo = 20
    learning_rate = 1e-1
    set_random_seeds(seed_value=2023, device=devices_name)

    print('======== Start Training ========')
    print(f'Learning rate: {learning_rate}')
    print('======== Standard VGG-A ========')
    model_Vgg = VGG_A()
    Opt = torch.optim.SGD(model_Vgg.parameters(), lr=learning_rate)
    Loss = nn.CrossEntropyLoss()
    tr_acc_BA, tr_val_BA = train(model_Vgg, Opt, Loss, train_loader, val_loader, epochs_n=epo)

    print('======== Start Training ========')
    print(f'Learning rate: {learning_rate}')
    print('======== VGG-A + BatchNorm ========')
    model_BN_Vgg = VGG_A_BatchNorm()
    Opt = torch.optim.SGD(model_BN_Vgg.parameters(), lr=learning_rate)
    Loss = nn.CrossEntropyLoss()
    tr_acc_BN, tr_val_BN = train(model_BN_Vgg, Opt, Loss, train_loader, val_loader, epochs_n=epo)

    Step = np.arange(0, tr_acc_BN.__len__(), 1, dtype=int) * 100

    # Plot
    plt.style.use('ggplot')
    plt.figure(figsize=(7, 5), dpi=800)
    plt.plot(Step, tr_acc_BA, 'g-', label='Training: Standard VGG', linewidth=1)
    plt.plot(Step, tr_val_BA, 'g:', label='Validation: Standard VGG', linewidth=1)
    plt.plot(Step, tr_acc_BN, color='firebrick', linestyle='-', linewidth=1,
             label='Training: Standard VGG + BachNorm')
    plt.plot(Step, tr_val_BN, color='firebrick', linestyle=':', linewidth=1,
             label='Validation: Standard VGG + BachNorm')
    plt.legend()
    plt.title('Learning Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Steps')
    plt.savefig('./results/BatchNorm_acc.png')
