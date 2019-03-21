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
    def __init__(self, inChannel, outChannel, stride = 1, leaky = False):
        super(bottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel//2, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(outChannel//2, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(outChannel//2, outChannel//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel//2, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(outChannel//2, outChannel, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)

        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.skipConv = nn.Conv2d(inChannel, outChannel, kernel_size=1, padding=0)
        self.skipBn = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)
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

        residual = self.skipConv(residual)
        residual = self.skipBn(residual)
        out += residual

        return out

    def saveWeight(self, filePointer):
    def loadWeight(self, weights, ptr):

class downSample2d(nn.Module):
    def __init__(self, inChannel, outChannel, stride = 2, leaky = False):
        super(downSample2d, self).__init__()
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

class upSample2d(nn.Module):
    def __init__(self):
        super(upSample2d, self).__init__()

class denseSequentialNet(nn.Module):
    def __init__(self, leaky=False, rnnType="GRU"):
        super(denseSequentialNet, self).__init__()

        # Build up two channels of encoder for the horizontal projection and
        # the perpendicular projection of RF signal energies.
        self.encoderX = self.buildEncoder()
        self.encoderY = self.buildEncoder()

        # Sequential processing layers, GRU, LSTM or QRNN, decided by input
        # parameter rnnType

        # Decoder receives a set of aligned characteristic vectors to predict
        # confidence map of spatial location of key points.
        self.leaky = leaky
        self.rnnType = rnnType
    def forward(self, x):
        return x

    def buildEncoder(self):
        layer = []

        # initial feature extraction layers
        layer.append(nn.BatchNorm2d(1, eps=0.001, momentum=0.01))
        layer.append(nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0))
        if self.leaky:
            layer.append(nn.LeakyReLU(inplace=True))
        else:
            layer.append(nn.ReLU(inplace=True))

        # first two bottleneck blocks
        layer.append(bottleNeck(32, 64, 1, self.leaky))
        layer.append(bottleNeck(64, 32, 1, self.leaky))

        # The 13x60 input go through 2 downsample layers and reach the dimension
        # of 2x8. Then align the tenser elements and get a characteristic vector.
        for i in range(2):
            layer.append(downSample2d(32*i, 64*i, self.leaky))
            layer.append(bottleNeck(64*i, 128*i, self.leaky))
            layer.append(bottleNeck(128*i, 64*i, self.leaky))

        return nn.ModuleList(layer)

    def buildRNN(self):
    def buildDecoder(self):

    def saveWeight(self, saveDirectory):
    def loadWeight(self, saveDirectory):