# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 5 21:15 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 5 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

__all__ = ['scaling', 'imwrite', 'sumup', 'sumup_perpendicular', 'sumup_horizontal']

import matplotlib.pyplot as plt
import numpy as np
import cv2

def scaling(img, img_size):
    '''
    Args:
         img         : (numpy.array) input image
         channels    : (int) number of input image channels
         img_size    : (int) dimensions of output image(square)
    Returns:
         Scaled image
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    # Getting new scales
    new_w = int(img_w * min(img_size/img_h, img_size/img_w))
    new_h = int(img_h * min(img_size/img_h, img_size/img_w))
    # Getting paddings
    pad_w = (img_size - new_w)//2
    pad_h = (img_size - new_h)//2
    # Resize
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    # Padding
    img = np.pad(img, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=128)

    return img


def imwrite(img, save_dir):
    '''
    Args:
         img         : (tensor) input image
         save_dir    : save directory
    '''
    img = img.squeeze(0).cpu().data.numpy()*255.0
    img = np.array(img, dtype=int).transpose((1, 2, 0))[:, :, ::-1]
    cv2.imwrite(save_dir, img)

def sumup(img):
    '''
    Args:
         img         : (np.array) heat maps to be sum up
    Returns:
         Adding all slices of heat map to a two-dimension matrix
    '''
    n = img.shape[0]
    sumimg = img[0 ,:, :]

    for i in range(1, n):
        sumimg = sumimg + img[i, :, :]

    #print(sumimg.shape)
    return sumimg

def sumup_perpendicular(img):
    '''
    Args:
         img          : (np.array) heat maps to sum up
    Returns:
         Adding all slices of heat map to two-dimension matrix perpendicularly
    '''
    n = img.shape[1]
    sumimg = img[:, 0, :]

    for i in range(1, n):
        sumimg = sumimg + img[:, i, :]

    return sumimg

def sumup_horizontal(img):
    '''
    Args:
         img          : (np.array) heat maps to sum up
    Returns:
         Adding all slices of heat map to two-dimension matrix perpendicularly
    '''
    n = img.shape[2]
    sumimg = img[:, :, 0]

    for i in range(1, n):
        sumimg = sumimg + img[:, :, i]

    return sumimg

