from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
# from ..functions.roi_align import RoIAlignFunction
from torchvision.ops import roi_align as RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(features, rois,
                                (self.aligned_height, self.aligned_width),
                                self.spatial_scale)


class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(features, rois,
                             (self.aligned_height, self.aligned_width),
                             self.spatial_scale)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(features, rois,
                             (self.aligned_height, self.aligned_width),
                             self.spatial_scale)
        return max_pool2d(x, kernel_size=2, stride=1)
