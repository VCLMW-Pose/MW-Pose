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
#   Shrowshoo-Young 2019-10, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from src.utils import scaling
from src.utils import putGaussianMap
from src.utils.imageproc import *
import matplotlib.pyplot as plt

import numpy as np
import torch
import cv2
import os

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
    def __init__(self, dataDirectory, signal_size = 60, GTSize = 64, imgw = 640, imgh = 360, valid = 0):
        '''
        :param dataDirectory: directory of saving dataset, the annotation
        txt is named as joint_point.txt.
        :param inputSize, inputDepth: dimensions of input RF signal heat maps.
        :param GTSize: dimensions of output confidence heat map of key points.
        :param imgw, imgh: dimensions of raw optical images, that is the
        reference dimension of key point annotation coordinates.
        :param clipFrame: clipFrame defines how many frames are included in
        each video clip. This does not mean deSeqNetLoader provides grouped
        data, but only single frames that is not concerned with clipFrame.
        This parameters instruct the reorder process, which ensures the images
        from the same video clip are grouped while reordering.
        :param selectPoint: this is a list defining what kinds of key points
        are wished to be predicted by DeSeqNet. Unselected key points will be
        absent in the ground truth confidence maps.
        :param paraparse: argparse class that keeps the following coefficients:
        sigma of 2d gaussian confidence map, expectation and sigma of rotation.
        :param rotate: defines whether to augment data through rotation.
        :param shuffle: defines whether reorder the sequence of data in each
        epoch.
        '''
        # Max number of key points
        self.MAX_POINTNUM = 18

        self.dataDirectory = dataDirectory
        # self.inputSize = inputSize
        # self.inputDepth = inputDepth
        self.GTSize = GTSize
        self.signal_size = signal_size

        # Read keypoint names
        with open(os.path.join(dataDirectory, 'keypoint.names'), 'r') as namefile:
            self.keyPointName = namefile.readlines()
            self.keyPointName = [line.rstrip() for line in self.keyPointName]

        # self.selectPoint = selectPoint
        # self.rotate = rotate
        # self.shuffle = shuffle
        # self.frames = clipFrame
        self.imgw = imgw
        self.imgh = imgh
        # self.paraparse = paraparse

        # Read annotation
        # self.anno = self.readAnnotation(dataDirectory)

        # Discard the redundant annotation that cannot make up a group
        if valid == 0:
            names = 'train.txt'
        else:
            names = 'valid.txt'

        with open(os.path.join(dataDirectory, names)) as namefile:
            self.names = namefile.readlines()
            self.names = [line.rstrip() for line in self.names]

        # remainder = length%self.frames
        # self.anno = self.anno[:length - remainder - 1]

        # reorder the groups randomly
        # if shuffle:
        #    self.reorder()

    def __getitem__(self, idx):
        '''
        Overide of __getitem__(). It acquire confidence map as ground truth
        and read raw RF signal matrix. If rotation operation is required, it
        performs rotation. The returns are confidence map and signal.
        '''
        # Get file name and eliminate its postfix '.jpg'
        file_name = self.names[idx]

        # Get confidence map ground truth and signal
        conf_maps = self.getGroundTruth(idx)
        signal = self.readSignal(file_name)

        # Data augmentation by rotating both signal and confidence map
        # if self.rotate:
        #    conf_maps, signal = self.rotation(conf_maps, signal)

        conf_maps = torch.from_numpy(conf_maps.astype(np.float32))
        signal = torch.from_numpy(signal.astype(np.float32)).div(signal.max())

        return conf_maps, signal

    def __len__(self):
        return len(self.names)

    def reorder(self):
        '''
        Shuffle the video clips. Each frame is divided into groups which is
        the least unit of this procedure.
        '''
        # Generate indexs of data groups
        groups = len(self.anno)/self.frames
        groupidx = np.arange(groups)

        # Shuffle the group index
        np.random.shuffle(groupidx)
        anno_new = []

        # Move the annotations to the new appointed address
        for i in groupidx:
            anno_new.append(self.anno[i*self.frames:(i + 1)*self.frames])

        self.anno = anno_new

    def rotation(self, conf_maps, signal):
        '''
        Data augmentation. Applied for RF signal matrix and confidence map
        '''
        # Generate gaussian distribution rotation angle
        sigma = self.paraparse.rotate_sigma
        rotate_ang = np.random.normal(0, sigma, 1)

        # Compute 2d rotation matrix and rotate RF signal
        M = cv2.getRotationMatrix2D(((self.inputSize - 1)/2, (self.inputSize - 1)/2), rotate_ang, 1)
        for i in range(self.inputDepth):
            signal[:, :, i] = cv2.warpAffine(signal[:, :, i], M, (self.inputSize, self.inputSize))

        # Compute 2d rotation matrix and rotate confidence map ground truth
        M = cv2.getRotationMatrix2D(((self.GTSize - 1)/2, (self.GTSize - 1)/2), rotate_ang, 1)
        for i in range(len(self.selectPoint)):
            conf_maps[:, :, i] = cv2.warpAffine(conf_maps[:, :, i], M, (self.GTSize, self.GTSize))

        return conf_maps, signal


    def getGroundTruth(self, idx):
        # Read annotation
        name = self.names[idx].split('/')[-1]
        annofile = open(os.path.join(self.dataDirectory, 'labels', name + '.txt'))
        anno = annofile.readlines()
        anno = [line.rstrip() for line in anno]

        # Allocate memory for confidence map
        conf_maps = np.zeros((self.MAX_POINTNUM, self.GTSize, self.GTSize))
        sigma = 3

        # Coefficient of coordinates transformation
        new_h = self.imgh*min(self.GTSize / self.imgh, self.GTSize / self.imgw)
        new_w = self.imgw*min(self.GTSize / self.imgh, self.GTSize / self.imgw)
        pad_h = (self.GTSize - new_h) / 2
        pad_w = (self.GTSize - new_w) / 2

        for idx, point_name in enumerate(self.keyPointName):
            coord = anno[idx]
            coord = coord.split(' ')
            coord = [int(coord[0]), int(coord[1])]
            if coord[0] == -1:
                continue

            # Transformation of coordinates
            coord[0] = coord[0]*(new_w / self.imgw)
            coord[1] = coord[1]*(new_h / self.imgw)
            coord[0] += pad_w
            coord[1] += pad_h

            # Add 2d gaussian confidence map
            conf_maps[idx, :, :] = putGaussianMap(coord, conf_maps[idx, :, :], sigma, [self.GTSize, self.GTSize])
            # heatmap = plt.pcolormesh(conf_maps[idx, :, :], cmap='jet')
            # plt.show()

        return conf_maps

    def readSignal(self, directory):
        # Open the directory in the form of read only binary file
        sig_file = open(directory, mode='rb')
        data = np.fromfile(sig_file, dtype=np.int32)

        # Dimensions
        try:
            size_x = data[0]
        except:
            print(directory)

        size_y = data[1]
        size_z = data[2]

        # Resize the signal as size_z x size_x x size_y
        raw_img = np.array(data[3:]).reshape(size_z, size_x, size_y)

        horizontal = np.zeros([60, 60])
        vertical = np.zeros([60, 60])
        horizontal[:, 23:36] = sumup_horizontal(raw_img)
        vertical[:, 23:36] = sumup_perpendicular(raw_img)

        # Expand dimension
        horizontal = np.expand_dims(horizontal, 0)
        horizontal = np.expand_dims(horizontal, 0)
        vertical = np.expand_dims(vertical, 0)
        vertical = np.expand_dims(vertical, 0)
        signal = np.concatenate((horizontal, vertical), 1)
        return signal

    def readAnnotation(self, dataDirectory):
        with open(os.path.join(dataDirectory, 'joint_point.txt'), 'r') as file:
            lines = file.readlines()

        # Annotation list
        anno = []
        for line in lines:
            # dictionary index to key point coordinates and file names
            person = {}
            line = line.split()
            fileName = line[0]
            person["fname"] = fileName

            for i in range(self.MAX_POINTNUM):
                # Read coordinates in the form of strings
                xcoord = line[2 + 2*i]
                ycoord = line[3 + 2*i]

                # Extract integers from string
                xcoord = int(xcoord.split('(')[1].rstrip(','))
                ycoord = int(ycoord.rstrip(')'))

                # Transfer to numpy array and assign to annotation recorder
                coord = np.array([xcoord, ycoord])
                person[self.keyPointName[i + 1]] = coord
            # Append to annotation list
            anno.append(person)

        return anno
