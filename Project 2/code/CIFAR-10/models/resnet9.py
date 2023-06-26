import torch.nn as nn


def conv_bn_act(c_in, c_out, kernel_size=(3, 3), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        # nn.RELU(),
        # nn.CELU(alpha=0.3),
    )


def conv_bn_act_pool(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        # nn.RELU(),
        # nn.CELU(alpha=0.3),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class resnet9(nn.Module):
    def __init__(self, c_in, num_classes=10):
        super().__init__()

        self.conv1 = conv_bn_act(c_in, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = conv_bn_act_pool(64, 128)
        self.conv3 = conv_bn_act(128, 128)
        self.conv4 = conv_bn_act(128, 128)
        self.conv5 = conv_bn_act_pool(128, 256)
        self.conv6 = conv_bn_act_pool(256, 512)
        self.conv7 = conv_bn_act(512, 512)
        self.conv8 = conv_bn_act(512, 512)
        self.pool9 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.linear10 = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        x = self.conv5(x)
        x = self.conv6(x)
        x = x + self.conv8(self.conv7(x))
        x = self.pool9(x)
        x = x.reshape(x.size(0), x.size(1))
        x = self.linear10(x)
        return x


def ResNet9():
    return resnet9(3)
