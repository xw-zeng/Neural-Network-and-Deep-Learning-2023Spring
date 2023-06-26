import sys
sys.path.append('E:/复旦/大三下/0神经网络与深度学习/Project/Finalpj')
from DessiLBI.ADAM_code.slbi_toolbox import SLBI_ToolBox_ADAM
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
    plt.savefig('./results' + savename + '.png')

sys.stdout = open('./log_prune.txt', 'w')
cudnn.benchmark = True
load_pth = torch.load('lenet.pth')
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
test_loader = load_data(dataset='MNIST', train=False, download=True, batch_size=64, shuffle=False)

# test prune two layers

print('pruning experiment')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('acc before pruning')
evaluate_batch(model, test_loader, 'cuda')

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('acc after pruning')

for ratio in [10, 20, 40, 60, 80]:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'pruning ratio: {ratio}%')
    optimizer.prune_layer_by_order_by_list(ratio, ['conv3.weight'], True)
    evaluate_batch(model, test_loader, 'cuda')
