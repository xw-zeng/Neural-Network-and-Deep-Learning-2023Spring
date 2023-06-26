import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import os, sys
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import time

from classification_model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
# epoch = 200
class_num = 10

class Trainer:
    def __init__(
        self, 
        model, 
        criterion_type, 
        optimizer_type, 
        regularization_mode: bool, 
        train_set,
        train_labels, 
        class_num, 
        batch_size,
        epoch_num,
    ):
        # Training parameters
        self.train_set = train_set
        self.train_labels = train_labels
        self.class_num = class_num 
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        
        # Hyper parameters
        self.lr = 0.0001
        self.momentum = 0.9
        self.SGD_weightdecay = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.Adam_weightdecay = 1e-4
        
        # Debug information
        self.print_step = 1000
        
        # 加载模型信息
        self.model = model
        # self.model = torch.load("./model/"+model_type+".pth")
        self.criterion = self.get_criterion(criterion_type)
        self.optimizer = self.get_optimizer(optimizer_type, regularization_mode)
        
        # weight decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma = 0.9)

        # 训练结束模型存储位置
        self.model_path = "./model/"+"MLP"+".pth"
        pass

        # result history
        self.training_acc_list = []
        self.training_loss_list = []
        self.testing_acc_list = []
        self.testing_loss_list = []
        self.best_training_acc = 0
        self.best_testing_acc = 0
        
    def train(self):
        iteration_num = 9000
        # self.model.to(device)
        for epoch in range(self.epoch_num):
            self.model.train(True)
            
            acc_cnt = 0
            epoch_loss = 0
            iteration_cnt = 0

            for i, data in enumerate(self.train_set):
                images = data
                labels = self.train_labels[i]
                images = images.to(device)
                labels = labels.to(device)

                output = self.model(images)
                
                true_labels = torch.zeros_like(input=output).to(device)
                # for row, label in enumerate(labels):
                true_labels[labels] = 1
                loss = self.criterion(output, true_labels)

                epoch_loss += loss.item()

                # update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predict_label = torch.max(output, -1)
                acc_cnt += torch.eq(predict_label, labels).cpu().sum()

                iteration_cnt += 1
                if iteration_cnt % self.print_step == 0:
                    acc = acc_cnt / iteration_cnt
                    print(f"Epoch [{epoch + 1} / {self.epoch_num}], Iteration [{iteration_cnt} / {iteration_num}], Loss Per Batch: {epoch_loss / (i + 1):.3f}, Acc: {acc:.3f}")
                    if (acc > self.best_training_acc):
                        self.best_training_acc, best_training_epoch, best_training_iteration = acc, epoch, iteration_cnt
            
            # acc = acc_cnt / iteration_num
            self.training_acc_list.append(acc)
            self.training_loss_list.append(epoch_loss)

            # save model
            torch.save(self.model, self.model_path)


        print(f"Best Training Acc: {self.best_training_acc:.3f}, in epoch [{best_training_epoch}], iteration [{best_training_iteration}]")
        # self.best_testing_acc, best_testing_epoch = torch.max(torch.tensor([self.testing_acc_list]), 1)
        # print(f"Best Testing Acc: {self.best_testing_acc:.3f}, in epoch [{best_testing_epoch}]")
        # torch.save(self.model, self.model_path)
    
    def result_display(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        x = torch.arange(0, self.epoch_num)
        
        ax1.plot(x, self.training_acc_list, 'b-')
        # ax1.plot(x, self.testing_acc_list, 'lightcoral-')

        ax2.plot(x, self.training_loss_list, 'r-')
        # ax2.plot(x, self.testing_loss_list, 'limegreen-')

        ax1.set_xlabel("Training Epoch")
        ax1.set_ylabel("Epoch Accuracy")
        ax2.set_ylabel("Epoch Loss")

        # plt.show()
        plt.savefig("./pic/" + "MLP" + "_acc_loss curve.png")
    
    def get_criterion(self, criterion_type):
        if criterion_type == 'MSE':
            criterion = nn.MSELoss()
        elif criterion_type == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss()
        return criterion
    
    def get_optimizer(self, optimizer_type, regularization_mode):
        if optimizer_type == 'SGD':
            if regularization_mode:
                optimizer = optim.SGD(list(self.model.parameters()), lr=self.lr, momentum=self.momentum, weight_decay=self.SGD_weightdecay)
            else:
                optimizer = optim.SGD(list(self.model.parameters()), lr=self.lr, momentum=self.momentum)
        elif optimizer_type == 'Adam':
            if regularization_mode:
                optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.Adam_weightdecay)
            else:
                optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
        return optimizer




if __name__ == "__main__":
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

    # 创建 MLP 模型实例
    input_size = 512  # 输入特征维度
    hidden_sizes = [1024, 1024, 1024]  # 隐藏层维度
    output_size = 10  # 输出维度
    model = MLP(input_size, hidden_sizes, output_size).to(device)
    
    # train_set, train_loader = load_training_set(modulate_list)
    trainer = Trainer(
        model=model,
        criterion_type="CrossEntropy",
        optimizer_type="Adam",
        regularization_mode=True,
        # train_loader=train_loader,
        train_set = train_set,
        train_labels = train_labels,
        class_num=class_num,
        batch_size=batch_size,
        epoch_num=50,
    )
    
    trainer.train()
    
    trainer.result_display()