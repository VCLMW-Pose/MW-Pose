# -*- coding: utf-8 -*-
'''
    Created on wed Sept 21 20:35 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 22 16:13 2018

South East University Automation College, 211189 Nanjing China
'''

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

__all__ = ['ResNet34']

class ResidualBlock(nn.Module):
    def __init__(self, chan_in, chan_out, stride=1, shortcut=None):
        '''
            Args:
                 chan_in       : (int) in channels
                 chan_out      : (int) out channels
                 stride        : (int) convolution stride
                 shortcut      : (nn.Module) shortcut module
        '''
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(chan_in ,chan_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan_out, chan_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chan_out)
        )

        self.right = shortcut

    def forward(self, x):
        '''
            Args:
                 x              : (torch.FloatTensor\torch.cuda.FloatTensor) input tensor
        '''
        x_ = self.left(x)
        residual = x if self.right is None else self.right(x)
        x_ += residual

        return x_

class ResNet34(nn.Module):
    def __init__(self, num_class):
        '''
            Args:
                 num_class       : (int) classes number
        '''
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 1)

    def _make_layer(self, chan_in, chan_out, block_num, stride=1):
        '''
            Args:
                 chan_in         : (int) input channels
                 chan_out        : (int) output channels
                 block_num       : (int) number of residual blocks
                 stride          : (int) convolution stride
        '''
        # shortcut layer
        shortcut = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, 1, stride, bias=False),
            nn.BatchNorm2d(chan_out)
        )

        layers =[]
        layers.append(ResidualBlock(chan_in, chan_out, stride, shortcut))

        for i in range(block_num):
            layers.append(ResidualBlock(chan_out, chan_out))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''
            Args:
                 x              : (torch.FloatTensor\torch.cuda.FloatTensor)
        '''
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

if __name__ == '__main__':
    model = ResNet34(2)
    print(model)