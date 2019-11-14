'''
    Created on Thu Sep 5 22:46 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Mon Sep 23 00:13 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from src.model.resnet34 import ResNet34
from src.dataset import deSeqNetLoader
from src.utils.pose_decoder import *
from src.utils.evaluation import *


class VGGNet(nn.Module):

    def __init__(self, leaky=False):
        super(VGGNet, self).__init__()
        self.leaky = leaky
        self.layer1 = nn.Sequential(
            ConvReLU(1, 64, (3, 3, 3), 1, self.leaky),
            ConvReLU(64, 64, (3, 3, 3), 1, self.leaky),  # no change in size
            nn.MaxPool3d((2, 2, 2))  # 64x30x32x32
        )
        self.layer2 = nn.Sequential(
            ConvReLU(64, 128, (3, 3, 3), 1, self.leaky),
            ConvReLU(128, 128, (3, 3, 3), 1, self.leaky),
            nn.MaxPool3d((2, 2, 2))  # 128x15x16x16
        )
        self.layer3 = nn.Sequential(
            ConvReLU(128, 256, (3, 3, 3), 1, self.leaky),
            ConvReLU(256, 256, (3, 3, 3), 1, self.leaky),
            ConvReLU(256, 256, (3, 3, 3), 1, self.leaky),
            nn.MaxPool3d((3, 2, 2))  # 256x5x8x8
        )
        self.layer2 = nn.Sequential(
            ConvReLU(256, 512, (3, 3, 3), 1, self.leaky),
            ConvReLU(512, 512, (3, 3, 3), 1, self.leaky),
            ConvReLU(512, 512, (3, 3, 3), 1, self.leaky),
            nn.MaxPool3d((1, 2, 2))  # 512x5x4x4
        )
        self.layer4 = nn.Sequential(
            nn.Linear(512*5*4*4, 4096),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        out = self.layer1(x)

        return out


class ConvReLU(nn.Module):
    def __init__(self, inchannel, outchannel, kernel, padding, leaky):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv3d(inchannel, outchannel, kernel_size=kernel, padding=padding)
        if leaky:
            self.relu = nn.LeakyReLU(inplace=False)
        else:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out
