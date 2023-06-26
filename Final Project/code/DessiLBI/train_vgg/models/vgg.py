"""
VGG
"""
import numpy as np
from torch import nn
from VGG_BatchNorm.utils.nn import init_weights_


# ## Models implementation
def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n


class VGG_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG_block, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.features(x)
        return x


class VGG_A(nn.Module):
    """VGG_A model

    size of Linear layers is smaller since input assumed to be 32x32x3, instead of
    224x224x3
    """

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64,
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            VGG_block(128, 256),  # stage 3
            VGG_block(256, 512),  # stage 4
            VGG_block(512, 512)   # stage5
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)
#
#
# class VGG_A_Light(nn.Module):
#     def __init__(self, inp_ch=3, num_classes=10):
#         super().__init__()
#
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(in_channels=inp_ch, out_channels=16,
#                       kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.stage2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32,
#                       kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         '''
#         self.stage3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.stage4 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.stage5 = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         '''
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 8 * 8, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes))
#
#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         # x = self.stage3(x)
#         # x = self.stage4(x)
#         # x = self.stage5(x)
#         x = self.classifier(x.view(-1, 32 * 8 * 8))
#         return x
#
#
# class VGG_A_Dropout(nn.Module):
#     def __init__(self, inp_ch=3, num_classes=10):
#         super().__init__()
#
#         self.features = nn.Sequential(
#             # stage 1
#             nn.Conv2d(in_channels=inp_ch, out_channels=64,
#                       kernel_size=(3, 3), padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # stage 2
#             nn.Conv2d(in_channels=64, out_channels=128,
#                       kernel_size=(3, 3), padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             VGG_block(128, 256),  # stage 3
#             VGG_block(256, 512),  # stage 4
#             VGG_block(512, 512)  # stage5
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512 * 1 * 1, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, num_classes))
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(-1, 512 * 1 * 1))
#         return x


class VGG_block_norm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG_block_norm, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.features(x)
        return x


class VGG_A_BatchNorm(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            VGG_block_norm(128, 256),  # stage 3
            VGG_block_norm(256, 512),  # stage 4
            VGG_block_norm(512, 512))  # stage 5

        self.classifier = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


if __name__ == '__main__':
    pass
