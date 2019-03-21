#-*- coding = utf-8 -*-
"""
# Copyright (c) 2018-2019, Shrowshoo-Young All Right Reserved.
#
# This programme is developed as free software for the investigation of human
# pose estimation using RF signals. Redistribution or modification of it is
# allowed under the terms of the GNU General Public Licence published by the
# Free Software Foundation, either version 3 or later version.
#
# Redistribution and use in source or executable programme, with or without
# modification is permitted provided that the following conditions are met:
#
# 1. Redistribution in the form of source code with the copyright notice
#    above, the conditions and following disclaimer retained.
#
# 2. Redistribution in the form of executable programme must reproduce the
#    copyright notice, conditions and following disclaimer in the
#    documentation and\or other literal materials provided in the distribution.
#
# This is an unoptimized software designed to meet the requirements of the
# processing pipeline. No further technical support is guaranteed.
"""

##################################################################################
#                                   deseqnet.py
#
#   Definitions of dense connection sequential processing network.
#
#   Shrowshoo-Young 2019-2, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class bottleNeck(nn.Module):
    def __init__(self, inChannel, procChannel, stride = 1, leaky = False):
        super(bottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, procChannel, kernel_size=1, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(procChannel, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(procChannel, procChannel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(procChannel, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(procChannel, procChannel*4, kernel_size=1, padding=1)
        self.bn3 = nn.BatchNorm2d(procChannel*4, eps=0.001, momentum=0.01)

        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class downSample(nn.Module):
    def __init__(self, inChannel, outChannel, stride = 2, leaky = False):
        super(downSample, self).__init__()
        self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.stride = stride
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class denseSequentialNet(nn.Module):
    def __init__(self):
        super(denseSequentialNet, self).__init__()
    def forward(self, x):
        return x