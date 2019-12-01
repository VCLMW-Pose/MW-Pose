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

__all__ = ['DeSeqNetProj', 'saveWeight', 'loadWeight']

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
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) \
                or isinstance(layer, nn.ConvTranspose2d):
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
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) \
                or isinstance(layer, nn.ConvTranspose2d):
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
    def __init__(self, inChannel, outChannel, stride=1, leaky=False, dropout=0):
        super(bottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel//2, kernel_size=1, stride=stride, padding=0)
        self.drop1 = nn.Dropout2d(dropout)
        self.bn1 = nn.BatchNorm2d(outChannel//2, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(outChannel//2, outChannel//2, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout2d(dropout)
        self.bn2 = nn.BatchNorm2d(outChannel//2, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(outChannel//2, outChannel, kernel_size=1, padding=0)
        self.drop3 = nn.Dropout2d(dropout)
        self.bn3 = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)

        if leaky:
            self.relu1 = nn.LeakyReLU(inplace=False)
            self.relu2 = nn.LeakyReLU(inplace=False)
        else:
            self.relu1 = nn.ReLU(inplace=False)
            self.relu2 = nn.ReLU(inplace=False)

        self.skipConv = nn.Conv2d(inChannel, outChannel, kernel_size=1, padding=0)
        self.skipdrop = nn.Dropout2d(dropout)
        self.skipBn = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.drop2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.drop3(out)
        out = self.bn3(out)

        residual = self.skipConv(residual)
        residual = self.skipdrop(residual)
        residual = self.skipBn(residual)

        out = out + residual

        return out

class downSample2d(nn.Module):
    def __init__(self, inChannel, outChannel, stride=[2, 2], leaky=False, dropout=0):
        super(downSample2d, self).__init__()
        self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.drop = nn.Dropout2d(dropout)
        self.bn = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)

        if leaky:
            self.relu = nn.LeakyReLU(inplace=False)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.dropout = dropout
        self.stride = stride
    
        out = self.conv(x)
        out = self.drop(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

class upSample2d(nn.Module):
    def __init__(self, inChannel, outChannel, stride=2, leaky=False, dropout=0):
        super(upSample2d, self).__init__()
        self.convTran = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.drop = nn.Dropout2d(dropout)
        self.bn = nn.BatchNorm2d(outChannel, eps=0.001, momentum=0.01)

        if leaky:
            self.relu = nn.LeakyReLU(inplace=False)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.stride = stride

    def forward(self, x):
        out = self.convTran(x)
        out = self.drop(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DeSeqNetProj(nn.Module):
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
    def __init__(self, leaky=False, rnnType="GRU", concatenateNum=0):
        super(DeSeqNetProj, self).__init__()

        self.leaky = leaky
        self.rnnType = rnnType
        self.concatenateNum = concatenateNum

        # Build up two channels of encoder for the horizontal projection and
        # the perpendicular projection of RF signal energies.
        self.encoderX = self.buildEncoder()
        self.encoderY = self.buildEncoder()

        self.linear = nn.Linear(1024, 1024)
        # Sequential processing layers, GRU, LSTM or QRNN, decided by input
        # parameter rnnType
        # if (rnnType == "GRU"):
        #     self.rnns = [nn.GRU(32 if i == 0 else numHidden, numHidden if i == numRNN - 1 else 32, 1, dropout=0)
        #                 for i in range(numRNN)]
        # elif(rnnType == "LSTM"):
        #     self.rnns = [nn.LSTM(32 if i == 0 else numHidden, numHidden if i == numRNN - 1 else 32, 1, dropout=0)
        #                  for i in range(numRNN)]

        # Decoder receives a set of aligned characteristic vectors to predict
        # confidence map of spatial location of key points.
        self.decoder = self.buildDecoder()

    def forward(self, x):
        '''
        Forward propagation of DeSeqNet includes a series of shortcut channels.
        For each encoder, there are shortcut channels in each down
        :param x:
        '''
        horiProj = x[:, :, 0, :, :]
        perpProj = x[:, :, 1, :, :]

        # Forward propagation of encoderX
        outx = self.encoderX[0](horiProj)
        for i in range(1, 5): outx = self.encoderX[i](outx)

        # Downsample layers
        for i in range(1, 4):
            outx = self.encoderX[2 + 3*i](outx)
            # Copy tensor in skip channel
            route = outx

            outx = self.encoderX[3 + 3*i](outx)
            outx = self.encoderX[4 + 3*i](outx)
            outx = torch.cat((outx, route), 1)

        outx = self.encoderX[14](outx)
        outx = outx.view(outx.size(0), -1)

        # Forward propagation of encoderY
        outy = self.encoderY[0](perpProj)
        for i in range(1, 5): outy = self.encoderY[i](outy)

        # Downsample layers
        for i in range(1, 4):
            outy = self.encoderY[2 + 3 * i](outy)
            route = outy

            outy = self.encoderY[3 + 3 * i](outy)
            outy = self.encoderY[4 + 3 * i](outy)
            outy = torch.cat((outy, route), 1)

        outy = self.encoderY[14](outy)
        outy = outy.view(outy.size(0), -1)
#        outy.backward()

        # Concatenate characteristic vectors get by encoderX and encoderY
        # as the input vector to RNNs
        characVec = torch.cat((outx, outy), 1)
        # for rnn, hid in zip(self.rnns, hidden):
        #    characVec, hid = rnn(characVec, hid)
        characVec = self.linear(characVec)
        n = characVec.shape[0]
        characVec_ = torch.zeros([n, 256, 2, 2]).cuda()

        for i in range(0, n):
            characVec_[i, :, :, :] = characVec[i, :].reshape([256, 2, 2])
        # Decoder propagation pipeline
        out = self.decoder[0](characVec_)

        # Upsample layers
        for i in range(1, 4):
            route = out
            out = self.decoder[3*i - 2](out)
            out = self.decoder[3*i - 1](out)

            out = torch.cat((out, route), 1)
            out = self.decoder[3*i](out)

        for i in range(10, 15):
            out = self.decoder[i](out)
        return out

    def buildEncoder(self):
        layer = []

        # initial feature extraction layers
        layer.append(nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0))
        layer.append(nn.BatchNorm2d(16, eps=0.001, momentum=0.01))
        if self.leaky:
            layer.append(nn.LeakyReLU(inplace=False))
        else:
            layer.append(nn.ReLU(inplace=False))

        # first two bottleneck blocks
        layer.append(bottleNeck(16, 32, 1, leaky = self.leaky))
        layer.append(bottleNeck(32, 32, 1, leaky = self.leaky))

        # The 13x60 input go through 2 downsample layers and reach the dimension
        # of 2x8. Then align the tenser elements and get a characteristic vector.
        for i in range(1, 3):
            layer.append(downSample2d(int(32*2**(i - 1)), int(32*2**(i - 1)), leaky = self.leaky))
            layer.append(bottleNeck(int(32*2**(i - 1)), int(64*2**(i - 1)), leaky = self.leaky))
            layer.append(bottleNeck(int(64*2**(i - 1)), int(32*2**(i - 1)), leaky = self.leaky))

        layer.append(downSample2d(128, 128, leaky = self.leaky))
        layer.append(bottleNeck(128, 256, leaky = self.leaky))
        layer.append(bottleNeck(256, 128, leaky = self.leaky))
        layer.append(downSample2d(256, 32, leaky = self.leaky))

        return nn.ModuleList(layer)

    def buildDecoder(self):
        # Decoder/generator model of keypoint confidence map
        layer = []

        # First two upsample layers
        layer.append(upSample2d(256, 64, leaky = self.leaky))
        layer.append(bottleNeck(64, 128, leaky = self.leaky))
        layer.append(bottleNeck(128, 64, leaky = self.leaky))
        layer.append(upSample2d(128, 64, leaky = self.leaky))

        # Two upsample layers
        for i in range(1, 2):
            layer.append(bottleNeck(int(64/(2**(i - 1))), int(128/(2**(i - 1))), leaky = self.leaky))
            layer.append(bottleNeck(int(128/(2**(i - 1))), int(64/(2**(i - 1))), leaky = self.leaky))
            layer.append(upSample2d(int(128/(2**(i - 1))), int(32/(2**(i - 1))), leaky = self.leaky))

        layer.append(bottleNeck(32, 64, leaky = self.leaky))
        layer.append(bottleNeck(64, 32, leaky = self.leaky))
        layer.append(upSample2d(64, 32, leaky = self.leaky))
        layer.append(bottleNeck(32, 24, 1, leaky = self.leaky))
        layer.append(bottleNeck(24, 18, 1, leaky = self.leaky))
        layer.append(upSample2d(18, 18, leaky=self.leaky))

        if self.leaky:
            layer.append(nn.LeakyReLU(inplace=False))
        else:
            layer.append(nn.LeakyReLU(inplace=False))
        layer.append(nn.ConvTranspose2d(18, 18, kernel_size=1, stride=2, padding=0))
        layer.append(nn.BatchNorm2d(18, eps=0.001, momentum=0.01))

        return nn.ModuleList(layer)

    def saveWeight(self, saveDirectory):
        # Open save directory and name the weight file as DeSeqNet
        filePointer = open(os.path.join(saveDirectory, "DeSeqNet"), 'wb')
        saveWeight(self.encoderX, filePointer)
        saveWeight(self.encoderY, filePointer)

        # Save weight of RNN
        for modules in self.rnns:
            modules.bias.data.cpu().numpy().tofile(filePointer)
            modules._all_weights.data.cpu().numpy().tofile(filePointer)

        # Save weight of decoder
        saveWeight(self.decoder, filePointer)
        filePointer.close()

    def loadWeight(self, saveDirectory):
        # Open save directory
        filePointer = open(os.path.join(saveDirectory, "DeSeqNet"), 'rb')
        weights = np.fromfile(filePointer, dtype=np.float32)
        ptr = 0

        # Load weight of two encoders
        ptr = loadWeight(self.encoderX, weights, ptr)
        ptr = loadWeight(self.encoderY, weights, ptr)

        # Load weights of RNN
        for modules in self.rnns:
            numBias = modules.bias.numel()
            bias = torch.from_numpy(weights[ptr:ptr + numBias]).view_as(modules.bias)
            modules.bias.data.copy_(bias)
            ptr += numBias

            numWeights = modules._all_weights.numel()
            weight = torch.from_numpy(weights[ptr:ptr + numWeights]).view_as(modules._all_weights)
            modules._all_weights.data.copy_(weight)
            ptr += numWeights

        # Load weight of decoder
        loadWeight(self.decoder, weights, ptr)

    #def concatenateModel(self):

class DeSeqNetFull(nn.Module):
    '''
    DeSeqNetFull gets 3 dimensional input, expecting to be 60x13x13. Suggested
    by input parameters imgSize and inpChannel:
                        inpChannel x imgSize x imgSize.
    The feature extraction encoders encoderX and encoderY in
    DeSeqNetProj are replaced with one encoder, with number of filters and
    down sample procedure changed. DeSeqNetFull adapts similar blocks in
    encoder with DeSeqNetProj, and use RNN--LSTM or GRU to process the
    characteristic vectors. This model implement the decoder using same
    decoder model adapted by DeSeqNetProj.
    Concatenation of DeSeqNetFull takes prediction from previous DeSeqNetFull
    and original input image as input. Deeper concatenation uses the
    prediction from several previous DeSeqNetFull and original input image
    as input to refine ultimate prediction. Intermediate supervision is
    applied to each DeSeqNetFull extricating gradient vanishing.
    Tricks like weight decay and dropout are adopted to handle over fitting,
    since the model is trained on a tiny dataset.
    '''
    def __init__(self, num_hid, num_rnn, leaky=False, rnn_type="GRU", concat_num=0, dropout=0):
        '''

        :param num_hid:
        :param num_rnn:
        :param leaky:
        :param rnn_type:
        :param concat_num:
        :param weight_decay:
        :param dropout:
        '''
        self.num_hid = num_hid
        self.num_rnn = num_rnn
        self.leaky = leaky
        self.rnn_type = rnn_type
        self.concat_num = concat_num
        self.dropout = dropout

        self.encoder = self.buildDecoder()
        return

    def forward(self):

        return

    def buildEncoder(self):
        layer = []

        # initial feature extraction layers
        layer.append(nn.Conv2d(60, 60, kernel_size=1, stride=1, padding=0))
        layer.append(nn.Dropout2d(p=self.dropout))
        layer.append(nn.BatchNorm2d(60, eps=0.001, momentum=0.01))
        if self.leaky:
            layer.append(nn.LeakyReLU(inplace=True))
        else:
            layer.append(nn.ReLU(inplace=True)) 

        # first two bottleneck blocks
        layer.append(bottleNeck(60, 60, 1, self.leaky, self.dropout))
        layer.append(bottleNeck(60, 60, 1, self.leaky, self.dropout))

        # The 13x13x60 input go through 2 downsample layers and reach the dimension
        # of 2x2x8. Then align the tenser elements and get a characteristic vector.
        layer.append(downSample2d(60, 120, 2, self.leaky, self.dropout))
        layer.append(bottleNeck(120, 240, 1, self.leaky, self.dropout))
        layer.append(bottleNeck(240, 120, 1, self.leaky, self.dropout))

        # Shortcut layer, concatenate intermediate result.
        layer.append(downSample2d(240, 120, 2, self.leaky, self.dropout))
        layer.append(bottleNeck(120, 240, 1, self.leaky, self.dropout))
        layer.append(bottleNeck(240, 120, 1, self.leaky, self.dropout))

        # Down sample layer 3 and another concatenate intermediate result.
        layer.append(downSample2d(240, 120, 2, self.leaky, self.dropout))

        return nn.ModuleList(layer)

    def buildDecoder(self):
        # The characteristic vector is transformed to 2x2x256. The dimension of
        # confidence map predicted by decoder is 128x128x16. Which requires three
        # times of up sample procedure.
        layer = []

        # First upsample layer
        layer.append(upSample2d(256, 256, 2, self.leaky, self.dropout))
        layer.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        layer.append(nn.Dropout2d(p=self.dropout))
        layer.append(nn.BatchNorm2d(256, eps=0.001, momentum=0.01))
        if self.leaky:
            layer.append(nn.LeakyReLU(inplace=True))
        else:
            layer.append(nn.ReLU(inplace=True))

        # Dimension: 8x8x256
        layer.append(upSample2d(256, 128, 2, self.leaky, self.dropout))
        layer.append(bottleNeck(128, 256, 1, self.leaky, self.dropout))
        layer.append(bottleNeck(256, 128, 1, self.leaky, self.dropout))

        # Upsample layer and first shortcut layer, dimension: 16x16x256
        layer.append(upSample2d(256, 128, 2, self.leaky, self.dropout))
        layer.append(bottleNeck(128, 128, 1, self.leaky, self.dropout))
        layer.append(bottleNeck(128, 128, 1, self.leaky, self.dropout))

        # second shortcut layer
        layer.append(upSample2d(256, 128, 2, self.leaky, self.dropout))
        layer.append(bottleNeck(128, 128, 1, self.leaky, self.dropout))
        layer.append(bottleNeck(128, 256, 1, self.leaky, self.dropout))



        return

    def saveWeight(self):
        return

    def loadWeight(self):
        return

class QDeSeqNetFull(nn.Module):
    '''
<<<<<<< Updated upstream
    QDeSeqNetFull improves the sequential processing capability of DeSeqNet
    by adapting Quasi-Recurrent Neural Network as the basic build block.
    It takes the row RF signal as input but not the projection heat maps.
    '''

    def __init__(self, num_hid, num_rnn, leaky=False, rnn_type="GRU", concat_num=0,
                 weight_decay=0, dropout=0):
        '''

        :param num_hid:
        :param num_rnn:
        :param leaky:
        :param rnn_type:
        :param concat_num:
        :param weight_decay:
        :param dropout:
        '''
        return

    def forward(self):
        return

    def buildEncoder(self):
        return

    def buildDecoder(self):
        return

    def saveWeight(self):
        return

    def loadWeight(self):
        return

class DeSeqNetTest(nn.Module):
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
    def __init__(self, leaky=False, rnnType="GRU", concatenateNum=0):
        super(DeSeqNetTest, self).__init__()

        self.leaky = leaky
        self.rnnType = rnnType
        self.concatenateNum = concatenateNum

        # Build up two channels of encoder for the horizontal projection and
        # the perpendicular projection of RF signal energies.
        self.encoderX = ResNet34(512)
        self.encoderY = ResNet34(512)

        # self.linear = nn.Linear(1024, 1024)
        # Sequential processing layers, GRU, LSTM or QRNN, decided by input
        # parameter rnnType
        # if (rnnType == "GRU"):
        #     self.rnns = [nn.GRU(32 if i == 0 else numHidden, numHidden if i == numRNN - 1 else 32, 1, dropout=0)
        #                 for i in range(numRNN)]
        # elif(rnnType == "LSTM"):
        #     self.rnns = [nn.LSTM(32 if i == 0 else numHidden, numHidden if i == numRNN - 1 else 32, 1, dropout=0)
        #                  for i in range(numRNN)]

        # Decoder receives a set of aligned characteristic vectors to predict
        # confidence map of spatial location of key points.
        # self.decoder = self.buildDecoder()

        self.decoder = nn.Sequential(

            UpBlock(1024, 512, upsample=True),
            UpBlock(512, 256, upsample=True),
            UpBlock(256, 128, upsample=True),
            UpBlock(128, 64, upsample=True),
            UpBlock(64, 36, upsample=False),
            UpBlock(36, 18, upsample=False),
           # UpBlock(32, 18, upsample=False),
            nn.Conv2d(18, 18, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        '''
        Forward propagation of DeSeqNet includes a series of shortcut channels.
        For each encoder, there are shortcut channels in each down
        :param x:
        '''
        horiProj = x[:, :, 0, :, :]
        perpProj = x[:, :, 1, :, :]

        # Forward propagation of encoderX
        outx = self.encoderX(horiProj)

        # Forward propagation of encoderY
        outy = self.encoderY(perpProj)

        out = torch.cat((outx, outy), 1)
        out = self.decoder(out)

        return out

class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

if __name__ == "__main__":
    model = DeSeqNetTest().to("cuda")
    model.load_state_dict(torch.load("../../train/checkpoints/deseqnettest_490.pth"))
    model.eval()
    # Get dataloader
    dataset = deSeqNetLoader("../../data/captest", test=True, valid=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    for batch_i, (image, signal, GT) in enumerate(dataloader):
        signal = Variable(signal.to("cuda"))
        image = image.squeeze(0).numpy()
        start_time = time.time()
        outputs = model(signal)
        end_time = time.time()
        outputs = outputs.cpu().detach().numpy()
        print(np.where(outputs[0, 4, :, :] == outputs[0, 4, :, :].max()))
        output_np, output_list = decoder(outputs)
        # output_np = pose_decode(outputs)
        # eval_pckh(output_np, GT, 18, 1)
        black = np.zeros((360, 640, 3))
        black = black.astype(np.uint8)
        plot_skeleton(black, output_list, thick=2)
        plot_skeleton(image, GT, thick=2)
        final = np.concatenate((black, image), axis=0)
        cv2.namedWindow('Test')
        cv2.imshow('Test', final)
        while True:
            if cv2.waitKey(10) & 0xFF == ord('\r'):
               break