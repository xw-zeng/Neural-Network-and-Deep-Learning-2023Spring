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
        test_loader, 
        class_num
    ):
        # Testing Parameters
        self.test_loader = test_loader
        self.class_num = class_num 
        
        
        # 加载模型信息
        self.model = self.get_model(model_path)
        
        pass
        
    def test(self):
        acc_cnt = 0
        self.testing_acc_list = []

        self.model.train(False)
        self.model.eval()

        for i, data in enumerate(self.test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            output = self.model(images)

            # acc
            _, predict_label = torch.max(output, 1)
            acc_cnt += torch.eq(predict_label, labels).cpu().sum()
        
        final_acc = acc_cnt / (i + 1) / 16
        self.testing_acc_list.append(final_acc)
        print(f"Accuracy: {final_acc}")
    
    def get_model(self, model_path):
        model = torch.load(model_path)
        model.to(device)
        # 打印模型的结构
        # ts.summary(model, (3, 32, 32))
        return model 

if __name__ == '__main__':
    test_loader = load_testing_data(batch_size=batch_size)
    tester = Evaluater(
        model_path="./model/ResNet18_CD_PreActivation.pth",
        test_loader=test_loader,
        class_num=class_num
    )
    tester.test()
