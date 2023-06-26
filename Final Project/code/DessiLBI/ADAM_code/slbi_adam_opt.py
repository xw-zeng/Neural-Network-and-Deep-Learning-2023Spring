import copy
import math
from collections import OrderedDict
import numpy as np
import torch
from torch.optim.optimizer import Optimizer


def shrink(s_t, lam=1.0):
    # proximal mapping for 2-d weight(fc layer)
    gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
    return gamma_t


def shrink_group(ts, lam=1.0):
    """ shrinkage for 4-d weight(conv layer), group as out"""
    ts_reshape = torch.reshape(ts, (ts.shape[0], -1))
    ts_norm = torch.div(torch.norm(ts_reshape, 2, 1), 1) # math.sqrt(ts_reshape.shape[1]))
    # ts_norm = torch.div(torch.norm(ts_reshape, 2, 1), ts_reshape.shape[1])
    ts_shrink = torch.max(torch.zeros_like(ts_norm),
                          torch.ones_like(ts_norm) - torch.div(lam * torch.ones_like(ts_norm), ts_norm))
    ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape, 0, 1), ts_shrink), 0, 1)
    ts_return = torch.reshape(ts_return, ts.shape)
    return ts_return


def get_numel(x):
    n = 1
    for i in range(len(x)):
        n *= x[i]
    return n


class SLBI(Optimizer):

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, kappa=1, mu=100):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, kappa=kappa, mu=mu)
        print('*******************************************')
        for key in defaults:
            print(key, ' : ', defaults[key])
        print('*******************************************')
        super(SLBI, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SLBI, self).__setstate__(state)

    def assign_name(self, name_list):
        for group in self.param_groups:
            for ITER, p in enumerate(group['params']):
                param_state = self.state[p]
                param_state['name'] = name_list[ITER]

    def initialize_slbi(self, layer_list=None):
        if layer_list is None:
            pass
        else:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if param_state['name'] in layer_list:
                        param_state['z_buffer'] = torch.zeros_like(p.data)
                        param_state['gamma_buffer'] = torch.zeros_like(p.data)
                        param_state['avg_dz'] = torch.zeros_like(p.data)
                        param_state['sq_dz'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            mu = group['mu']
            kappa = group['kappa']
            lr_kappa = group['lr'] * group['kappa']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Calculate the Adam Update
                param_state = self.state[p]
                param_state['step'] += 1
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                step_size = math.sqrt(bias_correction2) / bias_correction1
                grad = p.grad.data

                # update something...
                if 'z_buffer' in param_state:
                    # Initialize the grad
                    grad.add_(p.data - param_state['gamma_buffer'], alpha=1 / mu)

                if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
                    grad.add_(weight_decay, p.data)

                if 'z_buffer' in param_state:
                    # initialize the grad_z
                    avg_z, sq_z = param_state['avg_dz'], param_state['sq_dz']
                    grad_z = (param_state['gamma_buffer'] - p.data) / mu
                    avg_z.mul_(beta1).add_(grad_z)
                    sq_z.mul_(beta2).addcmul_(grad_z, grad_z, value=1 - beta2)
                    denom_z = sq_z.sqrt().add_(group['eps'])
                    param_state['z_buffer'].addcdiv_(avg_z, denom_z, value=-step_size * group['lr'])
                    # update the gamma
                    if len(p.data.size()) == 2:  # For the linear layer case
                        param_state['gamma_buffer'] = kappa * shrink(param_state['z_buffer'], 1)
                    elif len(p.data.size()) == 4:  # For the conv layer case
                        param_state['gamma_buffer'] = kappa * shrink_group(param_state['z_buffer'], 1)
                    else:
                        pass

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(other=group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size * lr_kappa)

    def calculate_layer_residue(self, layer_name):
        diff = 0
        for group in self.param_groups:
            mu = group['mu']
            diff_i = 0
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    if 'gamma_buffer' in param_state:
                        diff_i = ((p.data - param_state['gamma_buffer']) * (
                                p.data - param_state['gamma_buffer'])).sum().item()
                    else:
                        pass
            diff += diff_i / (2 * mu)
        print('Residue of' + layer_name + ' : ', diff)

    def calculate_all_residue(self):
        diff = 0
        for group in self.param_groups:
            mu = group['mu']
            diff_i = 0
            for p in group['params']:
                param_state = self.state[p]
                if 'gamma_buffer' in param_state:
                    diff_i += ((p.data - param_state['gamma_buffer']) * (
                            p.data - param_state['gamma_buffer'])).sum().item()
            diff += diff_i / (2 * mu)
        print('Residue : ', diff)

    def get_gamma_state_dict(self):
        gamma_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'gamma_buffer' in param_state:
                    gamma_dict[param_state['name']] = param_state['gamma_buffer']
        return gamma_dict

    def get_z_state_dict(self):
        z_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state:
                    z_dict[param_state['name']] = param_state['z_buffer']
        return z_dict

    def save_gamma_state_dict(self, path):
        gamma_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'gamma_buffer' in param_state:
                    gamma_dict[param_state['name']] = param_state['gamma_buffer']
        torch.save(gamma_dict, path)

    def generate_strong_mask_dict(self):
        mask = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state and len(p.data.size()) == 4:
                    mask[param_state['name']] = torch.ne(param_state['gamma_buffer'], 0.0)
        return mask

    def generate_weak_mask_dict(self):
        mask = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state:
                    mask[param_state['name']] = 1 - torch.ne(param_state['gamma_buffer'], 0.0)
        return mask

    def load_mask(self, mask):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['name'] in mask.keys():
                    state['mask'] = mask[state['name']]

    def apply_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'mask' in state:
                    p.data = p.data * state['mask']

    def calculate_w_star_by_layer(self, layer_name):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state and param_state['name'] == layer_name:
                    if len(p.data.size()) == 2:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                    elif len(p.data.size()) == 4:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                    else:
                        pass
                else:
                    pass

    def calculate_all_w_star(self, use_sparse=False):
        N = 0
        N_s = 0
        torch.cuda.empty_cache()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                N += p.size().numel()
                if 'z_buffer' in param_state:
                    if len(p.data.size()) == 1:
                        param_state['w_star'] = p.data * (torch.ne(param_state['gamma_buffer'], 0.0)).float()
                        num_selected_units = torch.ne(param_state['w_star'], 0.0).sum().item()
                        N_s += num_selected_units
                        if use_sparse:
                            param_state['original_params'] = p.data.clone()
                            p.data = param_state['w_star']
                        # print('###############################################')
                        # print('max:', torch.max(param_state['w_star']))
                        # print('min:', torch.min(param_state['w_star']))
                        # print(' Sparsity of ' + param_state['name'])
                        # print('number of selected channels:', num_selected_units)
                        # print('number of all channels:', p.data.size()[0])
                        # print('ratio of selected channels:', num_selected_units / (p.data.size()[0]))

                    elif len(p.data.size()) == 2:
                        param_state['w_star'] = p.data * (torch.ne(param_state['gamma_buffer'], 0.0)).float()
                        num_selected_units = torch.ne(param_state['w_star'], 0.0).sum().item()
                        N_s += num_selected_units
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']
                        # print('###############################################')
                        # print('max:', torch.max(param_state['w_star']))
                        # print('min:', torch.min(param_state['w_star']))
                        # print(' Sparsity of ' + param_state['name'])
                        # print('number of selected weights:', num_selected_units)
                        # print('number of all weights:', p.data.size()[1] * p.data.size()[0])
                        # print('ratio of selected weights:', num_selected_units / (p.data.size()[0] * p.data.size()[1]))
                        # print('###############################################')

                    elif len(p.data.size()) == 4:
                        param_state['w_star'] = p.data * (torch.ne(param_state['gamma_buffer'], 0.0)).float()
                        num_selected_units = torch.ne(param_state['w_star'], 0.0).sum().item()
                        N_s += num_selected_units
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']
                        # print('###############################################')
                        # print(' Sparsity of ' + param_state['name'])
                        # print('max:', torch.max(param_state['w_star']))
                        # print('min:', torch.min(param_state['w_star']))
                        # print(' Sparsity of ' + param_state['name'])
                        # print('number of channle :', p.data.size()[0])
                        # print('number of selected weights:', num_selected_units)
                        # print('number of all weights:',
                        #       p.data.size()[0] * p.data.size()[1] * p.data.size()[2] * p.data.size()[3])
                        # print('ratio of selected weights:', num_selected_units / (
                        #         p.data.size()[0] * p.data.size()[1] * p.data.size()[2] * p.data.size()[3]))
                        # print('###############################################')
                else:
                    N_s += p.size().numel()

    def recover(self):
        # in use_w_star or prune_layer, params are changed. so using recover() can give params original value
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'original_params' in param_state:
                    p.data = param_state['original_params']

    def calculate_layer_residue(self, layer_name):
        diff = 0
        for group in self.param_groups:
            mu = group['mu']
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    if 'gamma_buffer' in param_state:
                        diff = ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
                    else:
                        pass
        diff /= (2*mu)
        print('Residue of' + layer_name + ' : ', diff)

    def calculate_all_residue(self):
        diff = 0
        for group in self.param_groups:
            mu = group['mu']
            for p in group['params']:
                param_state = self.state[p]
                if 'gamma_buffer' in param_state:
                    diff += ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
        diff /= (2*mu)
        print('Residue : ', diff)

    def shrink(self, s_t, lam):
        #proximal mapping for 2-d weight(fc layer)
        gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
        return gamma_t

    def shrink_group(self, ts):
        # shrinkage for 4-d weight(conv layer)
        ts_reshape = torch.reshape(ts,(ts.shape[0],-1))
        ts_norm = torch.norm(ts_reshape,2,1)
        ts_shrink = torch.max(torch.zeros_like(ts_norm),torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm),ts_norm))
        ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape,0,1),ts_shrink),0,1)
        ts_return = torch.reshape(ts_return,ts.shape)
        return ts_return

    def use_w_star(self):
        #use sparse params to replace original params
        self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'w_star' in param_state:
                    if len(p.data.size())==4:
                        ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
                        ts_norm = torch.norm(ts_reshape, 2, 1)
                        num_selected_filters = torch.sum(ts_norm != 0).item()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = param_state['w_star']
                        print('max:', torch.max(param_state['w_star']))
                        print('min:', torch.min(param_state['w_star']))
                        print('number of filters: ',  p.data.size()[0])
                        print('number of selected filter:', num_selected_filters )
                    elif len(p.data.size())==2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = param_state['w_star']
                        print('max:', torch.max(param_state['w_star']))
                        print('min:', torch.min(param_state['w_star']))
                        print('number of filters: ',  p.data.size()[0] * p.data.size()[1])
                        print('number of selected units:', num_selected_units )
                    else:
                        pass

    def calculate_proportion(self, layer_name):
        #self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'w_star' in param_state and param_state['name'] == layer_name:
                    #print(layer_name)
                    self.calculate_w_star_by_layer(layer_name)
                    if len(p.data.size())==4:
                        ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
                        ts_norm = torch.norm(ts_reshape, 2, 1)
                        num_selected_filters = torch.sum(ts_norm != 0).item()
                        return num_selected_filters/p.data.size()[0]
                    elif len(p.data.size())==2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        return num_selected_units/p.data.size()[0] * p.data.size()[1]
                    else:
                        pass

    def calculate_norm(self, layer_name):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    layer_norm = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1)
        return layer_norm.cpu().detach().numpy()

    def cal_prune_thre(self, percent, layer_name):
        # prune layer according to given percent and layer name
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] in layer_name and 'prune_order' in param_state:
                    print(param_state['name'])
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    print('Threshold : ', threshold)
        return threshold

    def update_prune_order(self, epoch):
        self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state:
                    if len(p.data.size())==4:
                        if 'epoch_record' not in param_state:
                            param_state['epoch_record'] = torch.zeros_like(p.data).add_(2000.0)
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
                            epoch_per_filer, _ = torch.min(torch.reshape(param_state['epoch_record'], (param_state['epoch_record'].shape[0], -1)), dim=1) 
                            param_state['prune_order'] = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1) - epoch_per_filer
                        else:
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            #print(mask.size())
                            #print(param_state['epoch_record'].size())
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
                            epoch_per_filer, _ = torch.min(torch.reshape(param_state['epoch_record'], (param_state['epoch_record'].shape[0], -1)), dim=1) 
                            param_state['prune_order'] = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1) - epoch_per_filer        
                    elif len(p.data.size()) == 2:
                        if 'epoch_record' not in param_state:
                            param_state['epoch_record'] = torch.zeros_like(p.data).add_(2000.0)
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
                            param_state['prune_order'] = torch.abs(p.data) - param_state['epoch_record']
                        else:
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
                            #param_state['prune_order'] = torch.abs(param_state['w_star']) - param_state['epoch_record']
                            param_state['prune_order'] = torch.abs(p.data) - param_state['epoch_record']
                    else:
                        pass

    def prune_layer_by_order_by_name(self, percent, layer_name, prune_bias):
        # prune layer according to given percent and layer name
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name']==layer_name and 'prune_order' in param_state:
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    if len(p.data.size())==4:
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data[threshold > param_state['prune_order'], :, :, :] = 0.0
                        if prune_bias:
                            for k in range(iter + 1, len(group['params'])):
                                p_n = group['params'][k]
                                param_state_n = self.state[p_n]
                                if param_state_n['name'] == layer_name.replace('weight', 'bias'):
                                    param_state_n['original_params'] = copy.deepcopy(p_n.data)
                                    p_n.data[threshold > param_state['prune_order']] = 0.0
                    elif len(p.data.size())==2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        mask = (torch.gt(param_state['prune_order'], threshold)).float()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = p.data * mask
                    else:
                        pass
                elif param_state['name'] in layer_name and 'prune_order' not in param_state:
                    print('Please Update Order First')
                else:
                    pass

    #剪枝
    def prune_layer_by_order_by_list(self, percent, layer_name, prune_bias):
        # prune layer according to given percent and layer name
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] in layer_name and 'prune_order' in param_state:
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    if len(p.data.size())==4:
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data[threshold > param_state['prune_order'], :, :, :] = 0.0
                        if prune_bias:
                            for k in range(iter + 1, len(group['params'])):
                                p_n = group['params'][k]
                                param_state_n = self.state[p_n]
                                if param_state_n['name'] ==  param_state['name'].replace('weight', 'bias'):
                                    param_state_n['original_params'] = copy.deepcopy(p_n.data)
                                    p_n.data[threshold > param_state['prune_order']] = 0.0
                    elif len(p.data.size())==2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        mask = (torch.gt(param_state['prune_order'], threshold)).float()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = p.data * mask
                    else:
                        pass
                elif param_state['name'] in layer_name and 'prune_order' not in param_state:
                    print('Please Update Order First')
                else:
                    pass
