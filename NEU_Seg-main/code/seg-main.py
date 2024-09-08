import os
import torch
import torchvision

import torch
import torch.utils
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 此部分放置从source中导入的组件
import data_reader as dr
import InterpolateLayer
import Trainer


def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


def get_net(devices):
    num_classes = 4  # 4个类别，背景+3个灰度值
    resnet18 = torchvision.models.resnet18(weights=None)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    resnet18.add_module('upsample', InterpolateLayer(size=(200, 200)))

    return resnet18


def main():
    train_iter, test_iter = dr.load_data(64)
    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, Trainer.try_all_gpus()

    net = get_net(devices)
    # net.add_module('final_conv', nn.Conv2d(120, num_classes, kernel_size=1))
    # net.add_module('upsample', InterpolateLayer(size=(200, 200)))

    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    Trainer.train(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


if __name__ == '__main__':
    main()
