import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)

        running_mean = torch.zeros(num_features * num_splits)
        running_var = torch.ones(num_features * num_splits)

        self.weight.requires_grad = False
        self.num_splits = num_splits
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)

    def train(self, mode=True):
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
        return super().train(mode)

    def forward(self, input):
        n, c, h, w = input.shape
        if self.training or not self.track_running_stats:
            # assert n % self.num_splits == 0, f"Batch size ({n}) must be divisible by num_splits ({self.num_splits}) of GhostBatchNorm"
            return F.batch_norm(
                input.view(-1, c * self.num_splits, h, w),
                self.running_mean,
                self.running_var,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(n, c, h, w)
        else:
            return F.batch_norm(
                input,
                self.running_mean[: self.num_features],
                self.running_var[: self.num_features],
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )


def conv_bn_act(c_in, c_out, kernel_size=(3, 3), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False),
        GhostBatchNorm(c_out, num_splits=16),
        nn.GELU(),
        # nn.RELU(),
        # nn.CELU(alpha=0.3),
    )


def conv_pool_bn_act(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        GhostBatchNorm(c_out, num_splits=16),
        nn.GELU(),
        # nn.RELU(),
        # nn.CELU(alpha=0.3),
    )


class resnet9(nn.Module):
    def __init__(self, first_layer_weights, c_in, c_out, scale_out):
        super().__init__()
        c = first_layer_weights.size(0)
        conv1 = nn.Conv2d(c_in, c, kernel_size=(3, 3), padding=(1, 1), bias=False)
        conv1.weight.data = first_layer_weights
        conv1.weight.requires_grad = False

        self.conv1 = conv1
        self.conv2 = conv_bn_act(c, 64, kernel_size=(1, 1), padding=0)
        self.conv3 = conv_pool_bn_act(64, 128)
        self.conv4 = conv_bn_act(128, 128)
        self.conv5 = conv_bn_act(128, 128)
        self.conv6 = conv_pool_bn_act(128, 256)
        self.conv7 = conv_pool_bn_act(256, 512)
        self.conv8 = conv_bn_act(512, 512)
        self.conv9 = conv_bn_act(512, 512)
        self.pool10 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.linear11 = nn.Linear(512, c_out, bias=False)
        self.scale_out = scale_out

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv5(self.conv4(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = x + self.conv9(self.conv8(x))
        x = self.pool10(x)
        x = x.reshape(x.size(0), x.size(1))
        x = self.linear11(x)
        x = self.scale_out * x
        return x


def FastResNet9(weights, c_in, c_out, scale_out):
    return resnet9(weights, c_in, c_out, scale_out)
