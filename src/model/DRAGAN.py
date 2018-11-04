# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 3 21:39 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 3 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad
from src.utils.initialize import initialize_weight

class generator(nn.Module):
    def __init__(self, dim_in, dim_out, img_size):
        '''
        Args:
             dim_in       : (int) number of imput channels
             dim_out      : (int) number of output channels
             img_size     : (int) output image dimension
        '''
        super(generator, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.img_size = img_size

        self.fc = nn.Sequential(
            nn.Linear(self.dim_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128*(self.img_size//4)*(self.img_size//4)),
            nn.BatchNorm1d(128*(self.img_size//4)*(self.img_size//4)),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.dim_out, 4, 2, 1),
            nn.Tanh()
        )
        initialize_weight(self)

    def forward(self, x):
        '''
        Args:
             x           : (tensor) input tensor
        '''
        x = self.fc(x)
        x = x.view(-1, 128, self.img_size//4, self.img_size//4)
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    def __init__(self, dim_in, dim_out, img_size):
        '''
        Args:
             dim_in       : (int) number of imput channels
             dim_out      : (int) number of output channels
             img_size     : (int) input image dimension
        '''
        super(discriminator, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.img_size = img_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.dim_in, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*(self.img_size//4)*(self.img_size//4), 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.dim_out),
            nn.Sigmoid()
        )
        initialize_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*(self.img_size//4)*(self.img_size//4))
        x = self.fc(x)

        return x

class DRAGAN(nn.Module):
    def __init__(self, args):
        '''
        Args:

        '''



