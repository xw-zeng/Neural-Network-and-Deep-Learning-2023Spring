import sys
sys.path.append('E:/复旦/大三下/0神经网络与深度学习/Project/Finalpj')
from DessiLBI.ADAM_code.slbi_toolbox import SLBI_ToolBox_ADAM
from DessiLBI.code.slbi_toolbox import SLBI_ToolBox_Base
from utils import *
from data_loader import load_data
import lenet
from torch.backends import cudnn
import numpy as np
import matplotlib.pyplot as plt

def plot_weight_conv3(weight, title, savename):
    plt.figure(figsize=(5, 5), dpi=500)
    plt.clf()
    plt.imshow(weight, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig('./results/' + savename + '.png')

cudnn.benchmark = True
test_loader = load_data(dataset='MNIST', train=False, download=True, batch_size=64, shuffle=False)

load_pth = torch.load('lenet_base.pth')
torch.cuda.empty_cache()
model = lenet.Net().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox_Base(model.parameters(), lr=1e-2)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

print('BASE')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
evaluate_batch(model, test_loader, 'cuda')
# visulization
weight = model.conv3.weight.clone().detach().cpu().numpy()
H=10; W=10
weights = np.zeros((H*5, W*5))
for i in range(H):
    for j in range(W):
        weights[i*5:i*5+5, j*5:j*5+5] = weight[i][j]
weights = np.abs(weights)
plot_weight_conv3(weights, title='BASE: conv3 weight (DessiLBI)', savename='conv3_weight_base_dessilbi')

load_pth = torch.load('lenet_adam.pth')
torch.cuda.empty_cache()
model = lenet.Net().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox_ADAM(model.parameters(), lr=1e-2)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

print('ADAM')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
evaluate_batch(model, test_loader, 'cuda')
# visulization
weight = model.conv3.weight.clone().detach().cpu().numpy()
H=10; W=10
weights = np.zeros((H*5, W*5))
for i in range(H):
    for j in range(W):
        weights[i*5:i*5+5, j*5:j*5+5] = weight[i][j]
weights = np.abs(weights)
plot_weight_conv3(weights, title='ADAM: conv3 weight (DessiLBI)', savename='conv3_weight_adam_dessilbi')
