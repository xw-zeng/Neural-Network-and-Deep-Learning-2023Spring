#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import torch

from DessiLBI.ADAM_code.slbi_adam_opt import *
import copy
import math


class SLBI_ToolBox_ADAM(SLBI):

    def calculate_sparsity(self):
        # self.calculate_all_w_star()
        sparsity = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state:
                    # print(layer_name)
                    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
                        sparsity_fc = (torch.ne(param_state['gamma_buffer'], 0.0)).sum().item()
                        total_fc = p.data.numel()
                        sparsity.append(float(sparsity_fc) / total_fc)
                    else:
                        pass
        return sparsity

    def calculate_path(self, layer_name):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    # print(layer_name)
                    if len(p.data.size()) == 4:
                        ts_reshape = torch.reshape(p.data * (torch.ne(param_state['gamma_buffer'], 0.0)).float(),
                                                   (p.data.shape[0], -1))
                        return torch.norm(ts_reshape, 2, 1)
                    elif len(p.data.size()) == 2:
                        ts_reshape = torch.flatten(p.data * (torch.ne(param_state['gamma_buffer'], 0.0)).float())
                        return torch.abs(ts_reshape)
                    else:
                        pass

    def calculate_norm(self, layer_name):
        """Calculate the norm of a certain layer"""
        layer_norm = None
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    layer_norm = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1)
        return layer_norm.cpu().detach().numpy()

    def cal_prune_thre(self, percent, layer_name):
        """prune layer according to given percent and layer name"""
        threshold = None
        for group in self.param_groups:
            for _, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] in layer_name and 'prune_order' in param_state:
                    print(param_state['name'])
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    print('Threshold : ', threshold)
        return threshold

    def calculate_mask(self, layer_name):
        self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'w_star' in param_state and param_state['name'] == layer_name:
                    if len(p.data.size()) == 4:
                        ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
                        ts_norm = torch.norm(ts_reshape, 2, 1)
                        mask = torch.ones_like(param_state['w_star'])
                        mask[ts_norm != 0, :, :, :] = 0
                        return mask
                    elif len(p.data.size()) == 2:
                        return torch.ones_like(param_state['w_star'])
                    # return torch.gt(torch.abs(param_state['w_star']), 0.0).float()
                    else:
                        pass

    # TODO: Modified (OK)
    def step_with_freeze(self, freeze=True):
        loss = None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            mu = group['mu']
            kappa = group['kappa']
            lr_kappa = group['lr'] * group['kappa']
            lr_gamma = group['lr']
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

                if 'z_buffer' in param_state:
                    # calculate the mask
                    if freeze:
                        mask = self.calculate_mask(param_state['name'])
                    else:
                        mask = torch.ones_like(grad)
                    # Initialize the grad with mask...
                    grad = p.grad.data + (p.data - param_state['gamma_buffer']) / mu
                    grad = grad * mask
                    # initialize the grad_z
                    exp_avg_dz, exp_avg_sq_dz = param_state['exp_avg_dz'], param_state['exp_avg_sq_dz']
                    grad_z = mask * (param_state['gamma_buffer'] - p.data) / mu
                    exp_avg_dz.mul_(beta1).add_(1 - beta1, grad_z)
                    exp_avg_sq_dz.mul_(beta2).addcmul_(1 - beta2, grad_z, grad_z)
                    denom_z = exp_avg_sq_dz.sqrt().add_(group['eps'])
                    param_state['z_buffer'].addcdiv_(-lr_gamma * step_size, exp_avg_dz, denom_z)
                    if len(p.data.size()) == 2:
                        param_state['gamma_buffer'] = kappa * shrink(param_state['z_buffer'], 1)
                    elif len(p.data.size()) == 4:
                        param_state['gamma_buffer'] = kappa * shrink_group(param_state['z_buffer'])
                    else:
                        pass
                # return to calculate the update on w
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(-step_size * lr_kappa, exp_avg, denom)

    def print_network(self):
        print('Printing Network')
        for group in self.param_groups:
            for _, p in enumerate(group['params']):
                param_state = self.state[p]
                print(param_state['name'], p.data.size())

    def generate_dict(self):
        net_dict = {}
        for group in self.param_groups:
            for ITER, p in enumerate(group['params']):
                param_state = self.state[p]
                s_name = param_state['name'].replace('module.', '')
                if s_name == 'conv1.weight':
                    net_dict['conv1.out'] = p.data.size()[0]
                elif s_name == 'fc.weight':
                    net_dict['fc.in'] = p.data.size()[1]
                elif len(p.data.size()) == 4:
                    n_name = param_state['name'].replace('module.', '')
                    n_name = n_name.replace('.weight', '')
                    print(n_name)
                    net_dict[n_name + '.in'] = p.data.size()[1]
                    net_dict[n_name + '.out'] = p.data.size()[0]
                else:
                    pass
        return net_dict

    def get_size(self, layer_name):
        for group in self.param_groups:
            for _, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    return p.data.size()
