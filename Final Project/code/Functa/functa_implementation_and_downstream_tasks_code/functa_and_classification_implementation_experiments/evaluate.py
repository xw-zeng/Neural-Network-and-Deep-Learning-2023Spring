# %%
import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import os, sys
import torchvision
import torchvision.transforms as transforms
# from matplotlib import pyplot as plt 
from typing import Any, Callable, List, Optional, Type, Union

# Helper functions
# import torchsummary as ts
import os
# 这里关闭了GPU方便调试
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可

# 从子模块导入方法类
from dataloader import load_testing_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
class_num = 10

class Evaluater:
    def __init__(
        self, 
        model_path, 
        # test_loader,
        test_set,
        test_labels, 
        class_num
    ):
        # Testing Parameters
        # self.test_loader = test_loader
        self.test_set = test_set
        self.test_labels = test_labels
        self.class_num = class_num 
        
        
        # 加载模型信息
        self.model = self.get_model(model_path)
        
        pass
        
    def test(self):
        acc_cnt = 0
        self.testing_acc_list = []

        self.model.train(False)
        self.model.eval()

        for i, data in enumerate(self.test_set):
            images = data
            labels = self.test_labels[i]
            images = images.to(device)
            labels = labels.to(device)

            output = self.model(images)

            # acc
            _, predict_label = torch.max(output, -1)
            acc_cnt += torch.eq(predict_label, labels).cpu().sum()
        
        final_acc = acc_cnt / (i + 1)
        self.testing_acc_list.append(final_acc)
        print(f"Accuracy: {final_acc}")
    
    def get_model(self, model_path):
        model = torch.load(model_path)
        model.to(device)
        # 打印模型的结构
        # ts.summary(model, (3, 32, 32))
        return model 

if __name__ == '__main__':
    # test_loader = load_testing_data(batch_size=batch_size)
    # load training history
    modulate_list = []
    modulate = []
    
    # load modulations
    with open("modulate.txt", "r") as file:
        cnt = 1
        for line in file:
            if (cnt - 1) % 104 == 0:
                modulate_list.append(modulate)
                modulate = []
            if line[0] == 'T':
                cnt = cnt + 1
                continue
            else:
                line = line.rstrip("\n ]),")
                line = line.lstrip(" [")
                line = line.strip().split(",")
                for item in line:
                    modulate.append(float(item))
                cnt = cnt + 1
    modulate_list = modulate_list[1:]
    
    # load corresponding labels
    labels = []
    with open("labels.txt", "r") as file:
        for line in file:
            line = line.rstrip("\n")
            labels.append(int(line))
    
    # split into training and testing set
    train_set = torch.tensor(modulate_list[: 9000])
    train_labels = torch.tensor(labels[: 9000])
    test_set = torch.tensor(modulate_list[9000: ])
    test_labels = torch.tensor(labels[9000: ])
    
    tester = Evaluater(
        model_path="./model/MLP.pth",
        # test_loader=test_loader,
        test_set=test_set,
        test_labels=test_labels,
        class_num=class_num
    )
    tester.test()
