import numpy as np
from PIL import Image
import math


def DataAugmentation(img, crop = False):
    degree = np.random.uniform(-20, 20)
    img = img.reshape((16, 16),order='F')
    radius = math.pi * degree / 180
    width, height = img.shape
    if not crop:
        X1 = math.ceil(abs(0.5 * height * math.cos(radius) + 0.5 * width * math.sin(radius)))
        X2 = math.ceil(abs(0.5 * height * math.cos(radius) - 0.5 * width * math.sin(radius)))
        Y1 = math.ceil(abs(-0.5 * height * math.sin(radius) + 0.5 * width * math.cos(radius)))
        Y2 = math.ceil(abs(-0.5 * height * math.sin(radius) - 0.5 * width * math.cos(radius)))
        dstwidth = int(2 * max(X1, X2)) + 1
        dstheight = int(2 * max(Y1, Y2)) + 1
    else:
        dstheight = height
        dstwidth = width
    img_new = np.ones((dstwidth, dstheight)) * np.median(img)
    for i in range(dstwidth):
        for j in range(dstheight):
            new_i = int(
                (i - 0.5 * dstwidth) * math.cos(radius) - (j - 0.5 * dstheight) * math.sin(radius) + 0.5 * width)
            new_j = int(
                (i - 0.5 * dstwidth) * math.sin(radius) + (j - 0.5 * dstheight) * math.cos(radius) + 0.5 * height)
            if new_i >= 0 and new_i < width and new_j >= 0 and new_j < height:
                img_new[i, j] = img[new_i, new_j]
    if not crop:
        img_new = 1 - np.asarray(Image.fromarray(np.uint8(img_new * 32)).resize((16, 16))) / 255
    return img_new.reshape((256, ), order = 'F')
