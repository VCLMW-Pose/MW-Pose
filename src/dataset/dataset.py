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
#                                   dataset.py
#
#   Dataloader implementation of all networks.
#
#   Shrowshoo-Young 2019-2, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.utils import file_names
from src.utils import scaling
from src.utils import Annotation
import numpy as np
import torch
import cv2

__all__ = ['deSeqNetLoader']

class deSeqNetLoader(Dataset):
    '''
    The data loader of DeSeqNet. The annotations are kept in a txt file and
    each row stands for a person. deSeqNetLoader reads the annotation.
    When extracting data from deSeqNetloader, it generates confidence maps
    of key points as Ground Truth of deSeqNetLoader output.
    ***The batch size denotes the frame number of a video clip, ensure
    the batch size selected will not includes data from different video clip
    which may trigger non convergence of RNN layers!***
    ***Do not set shuffle to True! RNN layers need an intact and continuous
    video clip to converge.***
    '''
    def __init__(self, dataDirectory, inputSize):
        self.anno = Annotation(dataDirectory)
        self.inputSize = inputSize

        self.keyPointName = ['None',  # To be compatible with the pre-annotation
                            'rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'pelv',
                             'thrx', 'neck', 'head', 'rwri', 'relb', 'rsho', 'lsho',
                             'lelb', 'lwri']


    def __getitem__(self, idx):
        '''
        Args:
             idx            : (int) required index of corresponding data
        Returns:
             Required image (tensor)
        '''
        img = np.array(cv2.imread(self.img_list[idx]), dtype=float)
        img = scaling(img, 20)
        # Transform from BGR to RGB, HWC to CHW
        img = torch.FloatTensor(img[:, :, ::-1].transpose((2, 0, 1)).copy()).div(255.0)

        return img

    def __len__(self):
        return len(self.img_list)