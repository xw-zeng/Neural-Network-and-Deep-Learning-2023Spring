import os

import paddle
import numpy as np
from paddle.vision.datasets import MNIST

import utils

class Mnist(paddle.io.Dataset):
    """
    Toy Demo
    """
    def __init__(self, split, transforms=None):
        """
        Init
        """
        super().__init__()
        self.transforms = transforms
        self.split = split

        assert split in ['train', 'test']

        samples = MNIST(backend='cv2', mode=split)

        x_load = np.stack([sample[0] for sample in samples])
        y_load = np.stack([sample[1] for sample in samples])[:, 0]

        self.x = x_load / 255.
        self.y = y_load

        pad_amount = ((0, 0), (2, 2), (2, 2))
        self.x = np.pad(self.x, pad_amount, 'constant')

        # let's get some shapes to understand what we loaded.
        print('shape of X: {}, y: {}'.format(self.x.shape, self.y.shape))

    def __getitem__(self, idx):
        """
        get item
        """
        img = self.x[idx, ..., None]
        label = self.y[idx]

        if self.transforms:
            img = self.transforms(img)

        coord = utils.get_coordinate_grid(img.shape[1]).astype("float32")
        return img, coord, label, idx

    def __len__(self):
        return len(self.x)