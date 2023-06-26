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
from matplotlib import pyplot as plt
import time

# Helper functions
# import torchsummary as ts
import os
# 这里关闭了GPU方便调试
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可

# 从子模块导入方法类
from model import Flatten, get_simple_full_connect_model, get_simple_cnn_model, get_cnn_max_pooling_model, get_cnn_dropout_model, get_cnn_batchnorm2d_model, VGG_A, VGG18, ResNet18, ResNet18_CD_PreActivation, Identity, Identity_D, ResNet18_New, VGG_A_NOBN
from dataloader import load_training_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
# epoch = 200
class_num = 10

class Trainer:
    def __init__(
        self, 
        model_type, 
        criterion_type, 
        optimizer_type, 
        regularization_mode: bool, 
        train_loader, 
        class_num, 
        batch_size,
        epoch_num,
    ):
        # Training parameters
        self.train_loader = train_loader
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
        self.print_step = 10000
        
        # 加载模型信息
        # self.model = self.get_model(model_type)
        self.model = torch.load("./model/"+model_type+".pth")
        self.model_type = model_type
        self.criterion = self.get_criterion(criterion_type)
        self.optimizer = self.get_optimizer(optimizer_type, regularization_mode)
        
        # weight decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma = 0.9)

        # 训练结束模型存储位置
        self.model_path = "./model/"+model_type+".pth"
        pass

        # result history
        self.training_acc_list = []
        self.training_loss_list = []
        self.testing_acc_list = []
        self.testing_loss_list = []
        self.best_training_acc = 0
        self.best_testing_acc = 0
        
    def train(self):
        iteration_num = len(self.train_loader) * self.batch_size
        # self.model.to(device)
        for epoch in range(self.epoch_num):
            self.model.train(True)

            # weight decay
            # self.scheduler.step()
            
            acc_cnt = 0
            epoch_loss = 0
            iteration_cnt = 0

            for i, data in enumerate(self.train_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                output = self.model(images)
                
                true_labels = torch.zeros_like(input=output).to(device)
                for row, label in enumerate(labels):
                    true_labels[row][label] = 1
                loss = self.criterion(output, true_labels)

                epoch_loss += loss.item()

                # update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predict_label = torch.max(output, 1)
                acc_cnt += torch.eq(predict_label, labels).cpu().sum()

                iteration_cnt += self.batch_size
                if iteration_cnt % self.print_step == 0:
                    acc = acc_cnt / iteration_cnt
                    print(f"Epoch [{epoch + 1} / {self.epoch_num}], Iteration [{iteration_cnt} / {iteration_num}], Loss Per Batch: {epoch_loss / (i + 1):.3f}, Acc: {acc:.3f}")
                    if (acc > self.best_training_acc):
                        self.best_training_acc, best_training_epoch, best_training_iteration = acc, epoch, iteration_cnt
            
            # acc = acc_cnt / iteration_num
            self.training_acc_list.append(acc)
            self.training_loss_list.append(epoch_loss)

            # save model
            if epoch % 10 == 0:
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
        plt.savefig("./pic/" + self.model_type + "_acc_loss curve.png")
    
    def intermediate_result_display(self, pretrained_model, img, channel_id = 0):
        output_final = pretrained_model.forward(img.reshape(1, 3, 32, 32))

        output_setup = pretrained_model.output_setup
        output_layer1 = pretrained_model.output_layer1
        output_layer2 = pretrained_model.output_layer2
        output_layer3 = pretrained_model.output_layer3
        output_layer4 = pretrained_model.output_layer4
        
        plt.figure()
        plt.subplot(1, 5, 1)
        plt.imshow(output_setup[0].permute(1, 2, 0)[:,:,channel_id].detach().numpy())
        plt.subplot(1, 5, 2)
        plt.imshow(output_layer1[0].permute(1, 2, 0)[:,:,channel_id].detach().numpy())
        plt.subplot(1, 5, 3)
        plt.imshow(output_layer2[0].permute(1, 2, 0)[:,:,channel_id].detach().numpy())
        plt.subplot(1, 5, 4)
        plt.imshow(output_layer3[0].permute(1, 2, 0)[:,:,channel_id].detach().numpy())
        plt.subplot(1, 5, 5)
        plt.imshow(output_layer4[0].permute(1, 2, 0)[:,:,channel_id].detach().numpy())
        # plt.show()
        plt.savefig("./pic/"+self.model_type + "_intermediate_result.png")

    def get_model(self, model_type):
        model = None
        # Model
        if model_type == 'VGG18':
            model = VGG18()
        elif model_type == 'ResNet18':
            model = ResNet18()
        elif model_type == 'ResNet18_New':
            model = ResNet18_New()
        elif model_type == 'ResNet18_CD_PreActivation':
            model = ResNet18_CD_PreActivation()
        elif model_type == 'fc_simple':
            hidden_layer_size = 1024
            model = get_simple_full_connect_model(hidden_layer_size)
        elif model_type == 'cnn_simple':
            hidden_layer_size = 1024
            model = get_simple_cnn_model(hidden_layer_size)
        elif model_type == 'cnn_maxpool':
            model = get_cnn_max_pooling_model()
        elif model_type == 'cnn_dropout':
            model = get_cnn_dropout_model()
        elif model_type == 'cnn_bn':
            model = get_cnn_batchnorm2d_model()
        elif model_type == 'VGG-A':
            model = VGG_A()
        elif model_type == 'VGG_A_NOBN':
            model = VGG_A_NOBN()
        model.to(device)
        # 打印模型的结构
        # ts.summary(model, (3, 32, 32))
        return model 
    
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


if __name__ == '__main__':
    train_set, train_loader = load_training_set()
    trainer = Trainer(
        # model_type="VGG18", 
        # model_type="ResNet18", 
        model_type="ResNet18_CD_PreActivation", 
        # model_type="ResNet18_New",
        # model_type = "VGG_A_NOBN", 
        criterion_type="CrossEntropy",
        optimizer_type="Adam",
        regularization_mode=True,
        train_loader=train_loader,
        class_num=class_num,
        batch_size=batch_size,
        epoch_num=1,
    )
    trainer.train()
    # trainer.model = torch.load("./model/ResNet18_New.pth")
    # trainer.model.to('cpu')
    trainer.result_display()
    # print(trainer.model)
    # trainer.intermediate_result_display(pretrained_model=trainer.model.to(device), img=train_set[7][0].to(device), channel_id=0)
    pretrained_model = trainer.model.to(device)
    img = train_set[7][0].to(device)
    channel_id = 63
    # output_final = pretrained_model.forward(img.reshape(1, 3, 32, 32))

    output_setup1 = pretrained_model.out_layer_setup1
    
    # output_layer1 = pretrained_model.out_layer1
    # output_layer2 = pretrained_model.out_layer2
    # output_layer3 = pretrained_model.out_layer3
    # output_layer4 = pretrained_model.out_layer4

    output_layer1 = pretrained_model.out_layer1_block2
    output_layer2 = pretrained_model.out_layer2_block2
    output_layer3 = pretrained_model.out_layer3_block2
    output_layer4 = pretrained_model.out_layer4_block2    
    
    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(output_setup1[0].permute(1, 2, 0)[:,:,channel_id].cpu().detach().numpy())
    plt.subplot(1, 5, 2)
    plt.imshow(output_layer1[0].permute(1, 2, 0)[:,:,channel_id].cpu().detach().numpy())
    plt.subplot(1, 5, 3)
    plt.imshow(output_layer2[0].permute(1, 2, 0)[:,:,channel_id].cpu().detach().numpy())
    plt.subplot(1, 5, 4)
    plt.imshow(output_layer3[0].permute(1, 2, 0)[:,:,channel_id].cpu().detach().numpy())
    plt.subplot(1, 5, 5)
    plt.imshow(output_layer4[0].permute(1, 2, 0)[:,:,channel_id].cpu().detach().numpy())
    # plt.show()
    plt.savefig("./pic/"+trainer.model_type + "_intermediate_result.png")



