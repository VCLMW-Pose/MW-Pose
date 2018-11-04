# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 3 21:39 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 3 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad
from src.utils import initialize, printer

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
        initialize.initialize_weight(self)

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
        initialize.initialize_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*(self.img_size//4)*(self.img_size//4))
        x = self.fc(x)

        return x

class DRAGAN(nn.Module):
    def __init__(self, save_dir, result_dir, model_name, gpu_mode, img_size, z_dim, x_dim, lrG, lrD, beta1, beta2):
        '''
        Args:
             save_dir       : (string) directory to save weights
             result_dir     : (string)
             gpu_mode       : (boolean)
             img_size       : (int)
             z_dim          : (int)
             x_dim          : (int)
             lrG            : (float)
             lrD            : (float)
             beta1          : (float)
             beta2          : (float)
        '''
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.model_name = model_name
        self.img_size = img_size

        self.G = generator(dim_in=z_dim, dim_out=x_dim, img_size=self.img_size)
        self.D = discriminator(dim_in=x_dim, dim_out=1, img_size=self.img_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        if gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print("---------------Network Architecture---------------")
        printer.print_network(self.G)
        printer.print_network(self.D)
        print("--------------------------------------------------")

    def forward(self, z):
        '''
        Args:
             z              : (tensor) input scalar
        '''
        return self.G(z)

    def save(self):
        '''
        Returns:
             Save weights of generator and discriminator to corresponding binary file
        '''
        # Saving generator weights
        with open(os.path.join(self.save_dir + self.model_name + '_G'), 'wb') as file:
            fp = file

        for layer in self.G.modules():
            # Only deconvolutional layers, linear layers and batch normalization layers need to save weights
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear):
                layer.bias.data.cpu().numpy().tofile(fp)
                layer.weight.data.cpu().numpy().tofile(fp)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.bias.data.cpu().numpy().tofile(fp)
                layer.weight.data.cpu().numpy().tofile(fp)
                layer.running_mean.data.cpu().numpy().tofile(fp)
                layer.running_var.data.cpu().numpy().tofile(fp)

        fp.close()
        # Saving discriminator weights
        with open(os.path.join(self.save_dir + self.model_name + '_D'), 'wb') as file:
            fp = file

        for layer in self.D.modules():
            # Only convolutional layers, linear layers and batch normalization layers need to save weights
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.bias.data.cpu().numpy().tofile(fp)
                layer.weight.data.cpu().numpy().tofile(fp)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.bias.data.cpu().numpy().tofile(fp)
                layer.weight.data.cpu().numpy().tofile(fp)
                layer.running_mean.data.cpu().numpy().tofile(fp)
                layer.running_var.data.cpu().numpy().tofile(fp)

        fp.close()

    #def load(self):




