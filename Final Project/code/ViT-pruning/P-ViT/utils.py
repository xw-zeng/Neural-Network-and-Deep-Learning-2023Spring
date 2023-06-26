# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import typing


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# Added function to move mask to cuda
def mask_to_device(mask, device):
    return Mask({k: v.to(device) for k, v in mask.items()})

# Adapted from https://github.com/facebookresearch/open_lth/blob/2ce732fe48abd5a80c10a153c45d397b048e980c/pruning/pruned_model.py
def apply_mask(model, mask, device=None):
    if not device: device = next(model.parameters()).device
    cuda_mask = mask_to_device(mask, device)
    for name, param in model.named_parameters():
        if name in cuda_mask:
            param.data *= cuda_mask[name]

# from https://github.com/facebookresearch/open_lth/blob/main/models/base.py
def get_prunable_layer_names(net):
    """A list of the names of Tensors of this model that are valid for pruning.
    By default, only the weights of convolutional and linear layers are prunable.
    """

    return [name + '.weight' for name, module in net.named_modules() if
            isinstance(module, torch.nn.modules.conv.Conv2d) or
            isinstance(module, torch.nn.modules.linear.Linear)]

def vectorize(state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a state dict into a single column Tensor in a repeatable way."""

    return torch.cat([state_dict[k].reshape(-1) for k in sorted(state_dict.keys())])


def unvectorize(vector: torch.Tensor, reference_state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a vector back into a state dict with the same shapes as reference state_dict."""

    if len(vector.shape) > 1: raise ValueError('vector has more than one dimension.')

    state_dict = {}
    for k in sorted(reference_state_dict.keys()):
        if vector.nelement() == 0: raise ValueError('Ran out of values.')

        size, shape = reference_state_dict[k].nelement(), reference_state_dict[k].shape
        this, vector = vector[:size], vector[size:]
        state_dict[k] = this.reshape(shape)

    if vector.nelement() > 0: raise ValueError('Excess values.')
    return state_dict


def perm(N, seed: int = None):
    """Generate a tensor with the numbers 0 through N-1 ordered randomly."""

    gen = torch.Generator()
    if seed is not None: gen.manual_seed(seed)
    perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
    return torch.argsort(perm)


def shuffle_tensor(tensor: torch.Tensor, seed: int = None):
    """Randomly shuffle the elements of a tensor."""

    shape = tensor.shape
    return tensor.reshape(-1)[perm(tensor.nelement(), seed=seed)].reshape(shape)


def shuffle_state_dict(state_dict: typing.Dict[str, torch.Tensor], seed: int = None):
    """Randomly shuffle each of the tensors in a state_dict."""

    output = {}
    for i, k in enumerate(sorted(state_dict.keys())):
        output[k] = shuffle_tensor(state_dict[k], seed=None if seed is None else seed+i)
    return output

# from https://github.com/facebookresearch/open_lth/blob/main/pruning/mask.py
class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model, layers_to_prune=None):
        mask = Mask()
        if not layers_to_prune: layers_to_prune = model.get_prunable_layer_names()
        for name in layers_to_prune:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}
        # return {k: v.to(device).numpy() # v.cpu().numpy()
        #         for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return
# try:
# 	_, term_width = os.popen('stty size', 'r').read().split()
# except:
# 	term_width = 80
# term_width = int(term_width)
term_width = 80


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f