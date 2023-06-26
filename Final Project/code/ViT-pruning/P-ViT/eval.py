import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from collections import OrderedDict

from vit import ViT, channel_selection
from vit_slim import ViT_slim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

model = ViT(
    image_size = 224,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
model = model.to(device)

#import saved model here
model_path = "checkpoint/vit-4-ckpt.t7"
checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['acc']
check = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['net'].items())
# check = checkpoint['net']
model.load_state_dict(check)
print("Checkpoint '{}' loaded".format(model_path))

total = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        total += m.indexes.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        size = m.indexes.data.shape[0]
        bn[index:(index+size)] = m.indexes.data.abs().clone()
        index += size

percent = 0.5 # pruned_ratio
y, i = torch.sort(bn)
thre_index = int(total * percent)
thre = y[thre_index]

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            _, top5_preds = torch.topk(outputs, k=5)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            top5_correct += (top5_preds - targets[:, np.newaxis] == 0).sum()

        print('Top 1 Acc: %.3f%% (%d/%d) | Top 5 Acc %.3f%% (%d/%d)' % 
              (100.*correct/total, correct, total, 100.*top5_correct/total, top5_correct, total))


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

print('=======Baseline=======')
print('before pruning:')

test(model)

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, channel_selection):
        # print(k)
        # print(m)
        if k in [16,40,64,88,112,136]:
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            thre_ = thre.clone()
            while (torch.sum(mask)%8 !=0):                       # heads
                thre_ = thre_ - 0.0001
                mask = weight_copy.gt(thre_).float().cuda()
        else:
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.indexes.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
        #     format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total
print('=======Pruning=======')
print(f'Pruned ratio: {pruned_ratio}')
print(cfg)

cfg_prune = []
for i in range(len(cfg)):
    if i%2!=0:
        cfg_prune.append([cfg[i-1],cfg[i]])

newmodel = ViT_slim(image_size = 224,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    cfg=cfg_prune)

newmodel.to(device)
# num_parameters = sum([param.nelement() for param in newmodel.parameters()])

newmodel_dict = newmodel.state_dict().copy()

i = 0
newdict = {}
for k,v in model.state_dict().items():
    if 'net1.0.weight' in k:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'net1.0.bias' in k:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'to_q' in k or 'to_k' in k or 'to_v' in k:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'net2.0.weight' in k:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1
    elif 'to_out.0.weight' in k:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1

    elif k in newmodel.state_dict():
        newdict[k] = v

newmodel_dict.update(newdict)
newmodel.load_state_dict(newmodel_dict)

torch.save(newmodel.state_dict(), 'pruned.pth')
print('after pruning: ')
test(newmodel)
