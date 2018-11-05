# -*- coding: utf-8 -*-
'''
    Created on Sun Nov 4 10:46 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Sun Nov 4 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

__all__ = ['print_network']

import torch
import torch.nn as nn
import numpy as np

def print_network(network):
    '''
    Args:
         network        : (nn.Module) input network
    '''
    num_params = 0
    for param in network.parameters():
        num_params += param.numel()

    print(network)
    print("Total parameters: %d" % num_params)