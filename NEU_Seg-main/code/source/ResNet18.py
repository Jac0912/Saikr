import torch.nn as nn
import torch.nn.functional as F

import InterpolateLayer
import PPM

# 残差块类，可使用1x1的卷积层来调整x以用于相加
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
    net.add_module('ppm',PPM.PPM(64))
    # net.add_module('resnet_block2', resnet_block(128, 128, 2))
    net.add_module('resnet_block3', resnet_block(128, 256, 2))
    net.add_module('resnet_block4', resnet_block(256, 512, 2))

    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('upsample', InterpolateLayer.InterpolateLayer(size=(200, 200)))

    # net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
    # net.add_module('fc', nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))

    return net
