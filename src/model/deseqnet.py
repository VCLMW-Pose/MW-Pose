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

def saveWeight(model, filePointer):
    '''
    Save weight of a model to a binary file. This is a cross-framework neural
    network weight saving technique, which enables identical models deployed by
    other deep learning framework to load the pre-trained coefficient from
    any other frameworks. The sequence of saving and loading is strictly
    defined to eliminate possible confusion of saving order, which can refer to
    the given statements.
    This routine is designed for encapsulated network blocks, in which there is
    only convolution layers and batch normalization layers that is with weights.
    E.g. Models with linear layer or de-convolution layer require further
    modification of this routine.
    :param model: model to be saved
    :param filePointer: The file pointer of saving directory
    '''
    # Traverse every layer of input model
    for layer in model.modules():
        # Only deconvolutional layers, linear layers and batch normalization
        # layers need to save weights
        if isinstance(layer, nn.Conv2d):
            layer.bias.data.cpu().numpy().tofile(filePointer)
            layer.weight.data.cpu().numpy().tofile(filePointer)

        # Save weight of batch normalization layers
        elif isinstance(layer, nn.BatchNorm2d):
            layer.bias.data.cpu().numpy().tofile(filePointer)
            layer.weight.data.cpu().numpy().tofile(filePointer)
            layer.running_mean.data.cpu().numpy().tofile(filePointer)
            layer.running_var.data.cpu().numpy().tofile(filePointer)

def loadWeight(model, weights, ptr):
    '''
    Brief introduction of routine loadWeight() can refer to saveWeight(). Their
    applicable situations are identical.
    :param model: target model to be loaded
    :param weights: numpy array, and the data type is float32, which is the
    value of weights read from saving directory.
    :param ptr: current reading position. E.g. ptr is initialized with a, if
    a nn.Conv2d layer has 2000 parameters then the ptr is assigned with a + 2000
    after the weights is loaded.
    '''
    # Traverse every layer of input model
    for layer in model.modules():
        # Load weights for convolutional and batch normalization layers
        if isinstance(layer, nn.ConvTranspose2d):
            numBias = layer.bias.numel()
            bias = torch.from_numpy(weights[ptr:ptr + numBias]).view_as(layer.bias)
            layer.bias.data.copy_(bias)
            ptr += numBias

            numWeight = layer.weight.numel()
            weight = torch.from_numpy(weights[ptr:ptr + numWeight]).view_as(layer.weight)
            layer.weight.data.copy_(weight)
            ptr += numWeight

        # The number of coefficient of weights and bias of batch normalization
        # layers is identical
        elif isinstance(layer, nn.BatchNorm2d):
            numBias = layer.bias.numel()
            bias = torch.from_numpy(weights[ptr:ptr + numBias]).view_as(layer.bias)
            layer.bias.data.copy_(bias)
            ptr += numBias

            weight = torch.from_numpy(weights[ptr:ptr + numBias]).view_as(layer.weight)
            layer.weight.data.copy_(weight)
            ptr += numBias

            runMean = torch.from_numpy(weights[ptr:ptr + numBias]).view_as(layer.running_mean)
            layer.running_mean.data.copy_(runMean)
            ptr += numBias

            runVariance = torch.from_numpy(weights[ptr:ptr + numBias]).view_as(layer.running_var)
            layer.running_var.data.copy_(runVariance)
            ptr += numBias

    return ptr

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
    def __init__(self, inChannel, outChannel, stride=2, leaky=False):
        super(upSample2d, self).__init__()
        self.convTran = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)

        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.convTran(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class denseSequentialNet(nn.Module):
    '''
    Densely connected sequential processing network is a feature extraction
    and prediction model with sequential input process capability. The basic
    block of DeSeqNet includes two encoders---encoderX and encoderY, one for
    the perpendicular energy projection another for the horizontal energy
    projection; one\several sequential processing layer---GRU, LSTM or QRNN
    whose input is the characteristic vectors extracted with encoders.
    Sequential process capability grant the model robustness to the missing
    of energy reflection from certain limbs and guarantee refined outputs;
    One decoder, which generate prediction confidence maps of keypoints.

    Further concatenation of the basic DeSeqNet model, more or less improve its
    accuracy with significantly increased computation. We select one of
    ReLU or LeakyReLU as activation function through the model.
    '''
    def __init__(self, numHidden, numRNN, leaky=False, rnnType="GRU", concatenateNum=0):
        super(denseSequentialNet, self).__init__()

        # Build up two channels of encoder for the horizontal projection and
        # the perpendicular projection of RF signal energies.
        self.encoderX = self.buildEncoder()
        self.encoderY = self.buildEncoder()

        # Sequential processing layers, GRU, LSTM or QRNN, decided by input
        # parameter rnnType
        if (rnnType == "GRU"):
            self.rnns = [nn.GRU(32 if i == 0 else numHidden, numHidden if i == numRNN - 1 else 32, 1, dropout=0)
                        for i in range(numRNN)]
        elif(rnnType == "LSTM"):
            self.rnns = [nn.LSTM(32 if i == 0 else numHidden, numHidden if i == numRNN - 1 else 32, 1, dropout=0)
                         for i in range(numRNN)]

        # Decoder receives a set of aligned characteristic vectors to predict
        # confidence map of spatial location of key points.
        self.decoder = self.buildDecoder()

        self.leaky = leaky
        self.rnnType = rnnType

    def forward(self, x):
        #X = self.encoderX(x[:, :, 0])
        #y = self.encoderY(x[:, :, 1])

        #characVector =
        #characVector = self.rnns(characVector)
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

    def buildDecoder(self):
        # Decoder/generator model of keypoint confidence map
        layer = []

        # Two upsample layers
        for i in range(2):
            layer.append(bottleNeck(128/i, 256/i, self.leaky))
            layer.append(bottleNeck(256/i, 128/i, self.leaky))
            layer.append(upSample2d(128/i, 64/i))

        layer.append(bottleNeck(64, 32, 1, self.leaky))
        layer.append(bottleNeck(32, 64, 1, self.leaky))

        if self.leaky:
            layer.append(nn.LeakyReLU(inplace=True))
        else:
            layer.append(nn.LeakyReLU(inplace=True))
        layer.append(nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1, padding=0))
        layer.append(nn.BatchNorm2d(1, eps=0.001, momentum=0.01))

        return nn.ModuleList(layer)

    #def saveWeight(self, saveDirectory):
    #def loadWeight(self, saveDirectory):
    #def concatenateModel(self):