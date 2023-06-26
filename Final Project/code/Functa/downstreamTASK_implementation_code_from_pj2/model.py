import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
from typing import Any, Callable, List, Optional, Type, Union

# Helper functions
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
# 两层全连接网络
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


# 默认可用的简单模型

def get_simple_full_connect_model(hidden_layer_size = 1024):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(3*32*32, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax()
    )
    return model

def get_simple_cnn_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16*32*32, 10)
    )
    return model

def get_cnn_max_pooling_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(16*16*16, 10)
    )
    return model

def get_cnn_dropout_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(16*16*16, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 10)
    )
    return model

def get_cnn_batchnorm2d_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(16*16*16, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 10)
    )
    return model

# %%
def conv3x3(input_channels: int, output_channels: int, stride: int = 1, padding: int = 1, bias: bool = False, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
    )

# %%
def conv1x1(input_channels: int, output_channels: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=1,
        stride=stride,
        bias=bias,
    )

# %%
def conv7x7(input_channels: int = 3, output_channels: int = 64, stride: int = 2, padding: int = 3, bias: bool = False) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
    )

# %%
class Block(nn.Module):
    """Basic block in VGG-18 without / with BN layer"""
    def __init__(self, input_channels, output_channels, stride = 1, bn = False) -> None:
        super().__init__()

        self.conv3_first = conv3x3(
            input_channels=input_channels,
            output_channels=output_channels,
            stride=stride,
            )
        self.conv3 = conv3x3(
            input_channels=output_channels,
            output_channels=output_channels,
            stride=stride
            )
        self.relu = nn.ReLU(inplace=True)
        self.is_bn = bn
        self.bn = nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv3_first(x)
        if (self.is_bn):
            out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.is_bn):
            out = self.bn(out)
        out = self.relu(out)
        return out 


class VGG_A(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc=nn.Sequential(
            Flatten(),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256,10)
        )
        pass
    
    def forward(self,x):
        x=self.conv(x)
        # x = x.view(-1, 512)
        x=self.fc(x)
        return x

# %%
class VGG18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_setup = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.layer_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )

        self.out_layer_setup = None
        self.out_layer1 = None
        self.out_layer2 = None
        self.out_layer3 = None
        self.out_layer4 = None
        self.out_layer_final = None
    
    def forward(self, x):
        self.out_layer_setup = self.layer_setup(x)
        self.out_layer1 = self.layer1(self.out_layer_setup)
        self.out_layer2 = self.layer2(self.out_layer1)
        self.out_layer3 = self.layer3(self.out_layer2)
        self.out_layer4 = self.layer4(self.out_layer3)
        self.out_layer_final = self.layer_final(self.out_layer4)
        return self.out_layer_final


# %%
class VGG_18_Plain(nn.Module):
    """VGG-18 without / with BN layers"""
    def __init__(
        self,
        class_num,
        bn = False,
    ) -> None:
        super().__init__()

        self.input_channels = [64, 128, 256, 512]
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = conv7x7(input_channels=3, output_channels=self.input_channels[0])
        self.bn = nn.BatchNorm2d(num_features=self.input_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=self.input_channels[3], out_features=class_num)
        self.flatten = torch.flatten

        self.output_setup = None
        self.output_layer1 = None
        self.output_layer2 = None
        self.output_layer3 = None
        self.output_layer4 = None
        self.output_final = None

        if bn:
            self.setup = nn.Sequential(
                self.conv7,
                self.bn,
                self.relu,
                self.maxpool,
            )
        else:
            self.setup = nn.Sequential(
                self.conv7,
                self.relu,
                self.maxpool,
            )

        self.layer1 = nn.Sequential(
            Block(input_channels=self.input_channels[0], output_channels=self.input_channels[0], bn=bn),
            Block(input_channels=self.input_channels[0], output_channels=self.input_channels[0], bn=bn),
        )

        self.layer2 = nn.Sequential(
            Block(input_channels=self.input_channels[0], output_channels=self.input_channels[1], stride=2, bn=bn),
            Block(input_channels=self.input_channels[1], output_channels=self.input_channels[1], bn=bn),
        )

        self.layer3 = nn.Sequential(
            Block(input_channels=self.input_channels[1], output_channels=self.input_channels[2], stride=2, bn=bn),
            Block(input_channels=self.input_channels[2], output_channels=self.input_channels[2], bn=bn),
        )

        self.layer4 = nn.Sequential(
            Block(input_channels=self.input_channels[2], output_channels=self.input_channels[3], stride=2),
            Block(input_channels=self.input_channels[3], output_channels=self.input_channels[3]),
        )

    def forward(self, x: Tensor) -> Tensor:
        self.output_setup = self.setup(x)
        self.output_layer1 = self.layer1(self.output_setup)
        self.output_layer2 = self.layer2(self.output_layer1)
        self.output_layer3 = self.layer3(self.output_layer2)
        self.output_layer4 = self.layer4(self.output_layer3)
        out = self.avgpool(self.output_layer4)
        out = self.flatten(out, 1)
        self.output_final = self.fc(out)
        return self.output_final

# %%
class Identity(nn.Module):
    def __init__(self, in_channels, expansion, stride) -> None:
        super().__init__()
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * expansion, kernel_size=1, stride=stride, padding=0),
        )
    
    def forward(self, x):
        out = self.identity(x)
        return out

# %%
# 将identity的下采样交给avgpool去做，避免出现1x1卷积和stride=2同时出现造成信息流失
class Identity_D(nn.Module):
    def __init__(self, in_channels, expansion, stride) -> None:
        super().__init__()
        self.identity = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1), 
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * expansion, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        out = self.identity(x)
        return out

# %%

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_setup = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )
        self.layer1_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )

        self.layer2_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )
        self.layer2_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        self.layer3_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )
        self.layer3_block2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )

        self.layer4_block1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )
        self.layer4_block2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )

        self.layer_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )

        self.relu = nn.ReLU(inplace=True)
    
        self.out_layer_setup = None
        self.out_layer1_block1 = None
        self.out_layer1_block2 = None
        self.out_layer2_block1 = None
        self.out_layer2_block2 = None
        self.out_layer3_block1 = None
        self.out_layer3_block2 = None
        self.out_layer4_block1 = None
        self.out_layer4_block2 = None
        self.out_layer_final = None


    def forward(self, x):
        self.out_layer_setup = self.layer_setup(x)
        
        self.out_layer1_block1 = self.layer1_block1(self.out_layer_setup) + Identity(in_channels=64, expansion=1, stride=1).forward(self.out_layer_setup)
        self.out_layer1_block1 = self.relu(self.out_layer1_block1)
        self.out_layer1_block2 = self.layer1_block2(self.out_layer1_block1) + Identity(64, 1, 1).forward(self.out_layer1_block1)
        self.out_layer1_block2 = self.relu(self.out_layer1_block2)
        
        self.out_layer2_block1 = self.layer2_block1(self.out_layer1_block2) + Identity(64, 2, 2).forward(self.out_layer1_block2)
        self.out_layer2_block1 = self.relu(self.out_layer2_block1)
        self.out_layer2_block2 = self.layer2_block2(self.out_layer2_block1) + Identity(128, 1, 1).forward(self.out_layer2_block1)
        self.out_layer2_block2 = self.relu(self.out_layer2_block2)
        
        self.out_layer3_block1 = self.layer3_block1(self.out_layer2_block2) + Identity(128, 2, 2).forward(self.out_layer2_block2)
        self.out_layer3_block1 = self.relu(self.out_layer3_block1)
        self.out_layer3_block2 = self.layer3_block2(self.out_layer3_block1) + Identity(256, 1, 1).forward(self.out_layer3_block1)
        self.out_layer3_block2 = self.relu(self.out_layer3_block2)
        
        self.out_layer4_block1 = self.layer4_block1(self.out_layer3_block2) + Identity(256, 2, 2).forward(self.out_layer3_block2)
        self.out_layer4_block1 = self.relu(self.out_layer4_block1)
        self.out_layer4_block2 = self.layer4_block2(self.out_layer4_block1) + Identity(512, 1, 1).forward(self.out_layer4_block1)
        self.out_layer4_block2 = self.relu(self.out_layer4_block2)
        
        self.out_layer_final = self.layer_final(self.out_layer4_block2)
        return self.out_layer_final

# %%

class ResNet18_New(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_setup = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )
        self.layer1_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )

        self.layer2_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )
        self.layer2_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        self.layer3_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )
        self.layer3_block2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )

        self.layer4_block1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )
        self.layer4_block2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )

        self.layer_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )

        self.relu = nn.ReLU(inplace=True)
    
        self.out_layer_setup = None
        self.out_layer1_block1 = None
        self.out_layer1_block2 = None
        self.out_layer2_block1 = None
        self.out_layer2_block2 = None
        self.out_layer3_block1 = None
        self.out_layer3_block2 = None
        self.out_layer4_block1 = None
        self.out_layer4_block2 = None
        self.out_layer_final = None

        self.identity11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        )
        self.identity12 = self.identity11

        self.identity21 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0),
        )
        self.identity22 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
        )

        self.identity31 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0),
        )
        self.identity32 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        )

        self.identity41 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0),
        )
        self.identity42 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        self.out_layer_setup = self.layer_setup(x)
        
        # self.out_layer1_block1 = self.layer1_block1(self.out_layer_setup) + Identity(in_channels=64, expansion=1, stride=1).forward(self.out_layer_setup)
        self.out_layer1_block1 = self.layer1_block1(self.out_layer_setup) + self.identity11(self.out_layer_setup)
        self.out_layer1_block1 = self.relu(self.out_layer1_block1)
        # self.out_layer1_block2 = self.layer1_block2(self.out_layer1_block1) + Identity(64, 1, 1).forward(self.out_layer1_block1)
        self.out_layer1_block2 = self.layer1_block2(self.out_layer1_block1) + self.identity12(self.out_layer1_block1)
        self.out_layer1_block2 = self.relu(self.out_layer1_block2)
        
        # self.out_layer2_block1 = self.layer2_block1(self.out_layer1_block2) + Identity(64, 2, 2).forward(self.out_layer1_block2)
        self.out_layer2_block1 = self.layer2_block1(self.out_layer1_block2) + self.identity21(self.out_layer1_block2)
        self.out_layer2_block1 = self.relu(self.out_layer2_block1)
        # self.out_layer2_block2 = self.layer2_block2(self.out_layer2_block1) + Identity(128, 1, 1).forward(self.out_layer2_block1)
        self.out_layer2_block2 = self.layer2_block2(self.out_layer2_block1) + self.identity22(self.out_layer2_block1)
        self.out_layer2_block2 = self.relu(self.out_layer2_block2)
        
        # self.out_layer3_block1 = self.layer3_block1(self.out_layer2_block2) + Identity(128, 2, 2).forward(self.out_layer2_block2)
        self.out_layer3_block1 = self.layer3_block1(self.out_layer2_block2) + self.identity31(self.out_layer2_block2)
        self.out_layer3_block1 = self.relu(self.out_layer3_block1)
        # self.out_layer3_block2 = self.layer3_block2(self.out_layer3_block1) + Identity(256, 1, 1).forward(self.out_layer3_block1)
        self.out_layer3_block2 = self.layer3_block2(self.out_layer3_block1) + self.identity32(self.out_layer3_block1)
        self.out_layer3_block2 = self.relu(self.out_layer3_block2)
        
        # self.out_layer4_block1 = self.layer4_block1(self.out_layer3_block2) + Identity(256, 2, 2).forward(self.out_layer3_block2)
        self.out_layer4_block1 = self.layer4_block1(self.out_layer3_block2) + self.identity41(self.out_layer3_block2)
        self.out_layer4_block1 = self.relu(self.out_layer4_block1)
        # self.out_layer4_block2 = self.layer4_block2(self.out_layer4_block1) + Identity(512, 1, 1).forward(self.out_layer4_block1)
        self.out_layer4_block2 = self.layer4_block2(self.out_layer4_block1) + self.identity42(self.out_layer4_block1)
        self.out_layer4_block2 = self.relu(self.out_layer4_block2)
        
        self.out_layer_final = self.layer_final(self.out_layer4_block2)
        return self.out_layer_final


# %%

class ResNet18_CD_PreActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 将输入部分的1个7x7大卷积核换成3个3x3小卷积核
        self.layer_setup = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 预激活：先激活，再与identity相加
        self.layer1_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer3_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3_block2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.layer4_block1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer4_block2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.layer_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )

        self.relu = nn.ReLU(inplace=True)
    
        self.out_layer_setup = None
        self.out_layer1_block1 = None
        self.out_layer1_block2 = None
        self.out_layer2_block1 = None
        self.out_layer2_block2 = None
        self.out_layer3_block1 = None
        self.out_layer3_block2 = None
        self.out_layer4_block1 = None
        self.out_layer4_block2 = None
        self.out_layer_final = None

        self.identity11 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        )
        self.identity12 = self.identity11

        self.identity21 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
        )
        self.identity22 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
        )

        self.identity31 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
        )
        self.identity32 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        )

        self.identity41 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0),
        )
        self.identity42 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x):
        self.out_layer_setup = self.layer_setup(x)
        
        # self.out_layer1_block1 = self.layer1_block1(self.out_layer_setup)
        self.out_layer1_block1 = self.layer1_block1(self.out_layer_setup) + self.identity11(self.out_layer_setup)
        # self.out_layer1_block2 = self.layer1_block2(self.out_layer1_block1) + Identity_D(64, 1, 1).forward(self.out_layer1_block1)
        self.out_layer1_block2 = self.layer1_block2(self.out_layer1_block1) + self.identity12(self.out_layer1_block1)
        self.out_layer2_block1 = self.layer2_block1(self.out_layer1_block2) + self.identity21(self.out_layer1_block2)
        self.out_layer2_block2 = self.layer2_block2(self.out_layer2_block1) + self.identity22(self.out_layer2_block1)
        # self.out_layer2_block1 = self.layer2_block1(self.out_layer1_block2) + Identity_D(64, 2, 2).forward(self.out_layer1_block2)
        # self.out_layer2_block2 = self.layer2_block2(self.out_layer2_block1) + Identity_D(128, 1, 1).forward(self.out_layer2_block1)
        self.out_layer3_block1 = self.layer3_block1(self.out_layer2_block2) + self.identity31(self.out_layer2_block2)
        self.out_layer3_block2 = self.layer3_block2(self.out_layer3_block1) + self.identity32(self.out_layer3_block1)
        # self.out_layer3_block1 = self.layer3_block1(self.out_layer2_block2) + Identity_D(128, 2, 2).forward(self.out_layer2_block2)
        # self.out_layer3_block2 = self.layer3_block2(self.out_layer3_block1) + Identity_D(256, 1, 1).forward(self.out_layer3_block1)
        self.out_layer4_block1 = self.layer4_block1(self.out_layer3_block2) + self.identity41(self.out_layer3_block2)
        self.out_layer4_block2 = self.layer4_block2(self.out_layer4_block1) + self.identity42(self.out_layer4_block1)
        # self.out_layer4_block1 = self.layer4_block1(self.out_layer3_block2) + Identity_D(256, 2, 2).forward(self.out_layer3_block2)
        # self.out_layer4_block2 = self.layer4_block2(self.out_layer4_block1) + Identity_D(512, 1, 1).forward(self.out_layer4_block1)
        
        self.out_layer_final = self.layer_final(self.out_layer4_block2)
        return self.out_layer_final

# %%
class Basic_ResidualBlock(nn.Module):
    """Residual Block in ResNet-18"""
    def __init__(self, input_channels, output_channels, stride = 1, bn = True) -> None:
        super().__init__()

        self.conv3_first = conv3x3(
            input_channels=input_channels,
            output_channels=output_channels,
            stride=stride,
        )
        self.conv3 = conv3x3(
            input_channels=output_channels,
            output_channels=output_channels,
            stride=1
        )
        self.identity = conv1x1(
            input_channels=input_channels,
            output_channels=output_channels,
            stride=stride,
        )
        self.is_bn = bn
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv3_first(x)
        if self.is_bn:
            out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.is_bn:
            out = self.bn(out)
        out += self.identity(x)
        out = self.relu(out)
        return out

# %%
class ResNet_18(nn.Module):
    """ResNet-18 implementation"""
    def __init__(self, class_num) -> None:
        super().__init__()

        self.input_channels = [64, 128, 256, 512]
        self.conv7 = conv7x7(
            input_channels=3,
            output_channels=self.input_channels[0],
        )
        self.bn = nn.BatchNorm2d(num_features=self.input_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=self.input_channels[3], out_features=class_num)
        self.flattern = torch.flatten

        self.output_setup = None
        self.output_layer1 = None
        self.output_layer2 = None
        self.output_layer3 = None
        self.output_layer4 = None
        self.output_final = None

        self.setup = nn.Sequential(
            self.conv7,
            self.bn,
            self.relu,
            self.maxpool,
        )
    
        self.layer1 = nn.Sequential(
            Basic_ResidualBlock(input_channels=self.input_channels[0], output_channels=self.input_channels[0]),
            Basic_ResidualBlock(input_channels=self.input_channels[0], output_channels=self.input_channels[0]),
        )

        self.layer2 = nn.Sequential(
            Basic_ResidualBlock(input_channels=self.input_channels[0], output_channels=self.input_channels[1], stride=2),
            Basic_ResidualBlock(input_channels=self.input_channels[1], output_channels=self.input_channels[1]),
        )

        self.layer3 = nn.Sequential(
            Basic_ResidualBlock(input_channels=self.input_channels[1], output_channels=self.input_channels[2], stride=2),
            Basic_ResidualBlock(input_channels=self.input_channels[2], output_channels=self.input_channels[2]),
        )

        self.layer4 = nn.Sequential(
            Basic_ResidualBlock(input_channels=self.input_channels[2], output_channels=self.input_channels[3], stride=2, bn=False),
            Basic_ResidualBlock(input_channels=self.input_channels[3], output_channels=self.input_channels[3], bn=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        self.output_setup = self.setup(x)
        self.output_layer1 = self.layer1(self.output_setup)
        self.output_layer2 = self.layer2(self.output_layer1)
        self.output_layer3 = self.layer3(self.output_layer2)
        self.output_layer4 = self.layer4(self.output_layer3)
        out = self.avgpool(self.output_layer4)
        out = self.flattern(out, 1)
        self.output_final = self.fc(out)
        return self.output_final
