import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class PPM(nn.Module):
    def __init__(self, in_channel, pool_size=[1, 2, 3, 6]):
        super(PPM, self).__init__()
        self.pool_size = pool_size
        self.out_channel = int(in_channel / len(pool_size))
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size) for output_size in pool_size])
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channel, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False) for _ in pool_size
        )
        self.bns = ModuleList([nn.BatchNorm2d(self.out_channel) for _ in pool_size])

    def forward(self, x):
        size = x.size()  # (batch_size, channels, height, width)
        features = [x]

        for pool, conv, bn in zip(self.pools, self.convs, self.bns):
            poolde = pool(x)
            poolde = conv(poolde)
            poolde = bn(poolde)
            print(poolde.size())
            poolde = F.interpolate(poolde, size=size[2:], mode='bilinear', align_corners=False)
            features.append(poolde)

        out = torch.cat(features, dim=1)
        return out


if __name__ == "__main__":
    ppm = PPM(in_channel=64)
    x = torch.randn(8, 64, 128, 128)
    output = ppm(x)
    print(output.shape)
