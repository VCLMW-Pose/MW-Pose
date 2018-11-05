# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 5 21:06 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 5 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

__all__ = ['file_names']

import os

def file_names(root_dir):
    '''
    Args:
         root_dir        : (string) directory of root file folder
    Returns:
         Target list, storing file names involved in root file folder
    '''
    names = []
    # Traverse every file in root file folder
    for list in os.listdir(root_dir):
        if list[-3:] == 'jpg' or list[-3:] == 'png':
            path = os.path.join(root_dir, list)
            names.append(path)

    return names

