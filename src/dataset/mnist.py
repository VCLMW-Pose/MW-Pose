# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 3 21:39 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 3 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# MNIST dataset definition
class MNIST(Dataset):
    def __init__(self, folder_path, img_size):
        '''
        Args:
             folder_path    : (string) directory storing MNIST data set
             img_size       : (int) input dimensions of MNIST images
        '''
