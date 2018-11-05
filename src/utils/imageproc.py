# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 5 21:15 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 5 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

__all__ = ['scaling']

import numpy as np
import cv2

def scaling(img, channels, img_size):
    '''
    Args:
         img         : (numpy.array) input image
         channels    : (int) number of input image channels
         img_size    : (int) dimensions of output image(square)
    Returns:
         Scaled image
    '''
    img_w, img_h = img.size[1], img.size[0]
    # Getting new scales
    new_w = img_w * min(img_size/img_h, img_size/img_w)
    new_h = img_h * min(img_size/img_h, img_size/img_w)
    # Getting paddings
    pad_w = (img_size - new_w)//2
    pad_h = (img_size - new_h)//2
    # Resize
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    # Padding
    img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=128)

    return img
