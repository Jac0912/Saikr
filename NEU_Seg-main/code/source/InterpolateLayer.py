
import torch.nn as nn
import torch.nn.functional as F

class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor=None, size=None, mode='bilinear', align_corners=False):
        super(InterpolateLayer, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode, align_corners=self.align_corners)