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
    ***Do not set shuffle of base class to True! RNN layers need an intact
    and continuous video clip to converge. If shuffle is needed, please set
    the input parameter shuffle of deSeqNetLoader to True.***
    '''
    def __init__(self, dataDirectory, inputSize, GTSize, clipFrame, selectPoint, rotate=False, shuffle=False):
        '''
        :param dataDirectory: directory of saving dataset, the annotation
        txt is named as joint_point.txt.
        :param inputSize: dimensions of input RF signal heat maps.
        :param GTSize: dimensions of output confidence heat map of key points.
        :param clipFrame: clipFrame defines how many frames are included in
        each video clip. This does not mean deSeqNetLoader provides grouped
        data, but only single frames that is not concerned with clipFrame.
        This parameters instruct the reorder process, which ensures the images
        from the same video clip are grouped while reordering.
        :param selectPoint: this is a list defining what kinds of key points
        are wished to be predicted by DeSeqNet. Unselected key points will be
        absent in the ground truth confidence maps.
        :param rotate: defines whether to augment data through rotation.
        :param shuffle: defines whether reorder the sequence of data in each
        epoch.
        '''
        self.anno = Annotation(dataDirectory)
        self.inputSize = inputSize
        self.GTSize = GTSize

        self.keyPointName = ['None',  # To be compatible with the pre-annotation
                            'rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'pelv',
                             'thrx', 'neck', 'head', 'rwri', 'relb', 'rsho', 'lsho',
                             'lelb', 'lwri']
        self.selectPoint = selectPoint
        self.rotate = rotate
        self.shuffle = shuffle
        self.frames = clipFrame

        if shuffle:
            self.reorder()

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

    def reorder(self):
        return

    def rotate(self):
        return

    def confidenceMap(self):
        return

    def getGroundTruth(self):
        return