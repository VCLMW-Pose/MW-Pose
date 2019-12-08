# -*- coding = utf-8 -*-


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

__all__ = ['VGGNetLoader']


class VGGNetLoader(Dataset):
    '''

    '''

    def __init__(self, dataDirectory, signal_size=60, GTSize=64, imgw=640, imgh=360, valid=0):
        '''

        '''
        # Max number of key points
        self.MAX_POINTNUM = 18

        self.dataDirectory = dataDirectory

        self.GTSize = GTSize

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
        self.signals = []
        self.GTs = []
        self.conf_maps = []

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
        for i, name in enumerate(self.names):

            # Get confidence map ground truth and signal
            conf_map, GT = self.getGroundTruth(i)
            signal = self.readSignal(name)

            self.conf_maps.append(torch.from_numpy(conf_map.astype(np.float32)))
            self.signals.append(torch.from_numpy(signal.astype(np.float32)).div(signal.max()))
            self.GTs.append(GT)




    def __getitem__(self, idx):
        '''
        Overide of __getitem__(). It acquire confidence map as ground truth
        and read raw RF signal matrix. If rotation operation is required, it
        performs rotation. The returns are confidence map and signal.
        '''
        return self.conf_maps[idx], self.signals[idx], self.GTs[idx].copy()

    def __len__(self):
        return len(self.names)

    def reorder(self):
        '''
        Shuffle the video clips. Each frame is divided into groups which is
        the least unit of this procedure.
        '''
        # Generate indexs of data groups
        groups = len(self.anno) / self.frames
        groupidx = np.arange(groups)

        # Shuffle the group index
        np.random.shuffle(groupidx)
        anno_new = []

        # Move the annotations to the new appointed address
        for i in groupidx:
            anno_new.append(self.anno[i * self.frames:(i + 1) * self.frames])

        self.anno = anno_new

    def rotation(self, conf_maps, signal):
        '''
        Data augmentation. Applied for RF signal matrix and confidence map
        '''
        # Generate gaussian distribution rotation angle
        sigma = self.paraparse.rotate_sigma
        rotate_ang = np.random.normal(0, sigma, 1)

        # Compute 2d rotation matrix and rotate RF signal
        M = cv2.getRotationMatrix2D(((self.inputSize - 1) / 2, (self.inputSize - 1) / 2), rotate_ang, 1)
        for i in range(self.inputDepth):
            signal[:, :, i] = cv2.warpAffine(signal[:, :, i], M, (self.inputSize, self.inputSize))

        # Compute 2d rotation matrix and rotate confidence map ground truth
        M = cv2.getRotationMatrix2D(((self.GTSize - 1) / 2, (self.GTSize - 1) / 2), rotate_ang, 1)
        for i in range(len(self.selectPoint)):
            conf_maps[:, :, i] = cv2.warpAffine(conf_maps[:, :, i], M, (self.GTSize, self.GTSize))

        return conf_maps, signal

    def getGroundTruth(self, idx):
        # Read annotation
        name = self.names[idx].split('/')[-1]
        annofile = open(os.path.join(self.dataDirectory, 'labels', name + '.txt'))
        anno = annofile.readlines()
        anno = [line.rstrip() for line in anno]
        GT = np.zeros([len(anno), 2])

        # Allocate memory for confidence map
        conf_maps = np.zeros((self.MAX_POINTNUM, self.GTSize, self.GTSize))
        sigma = 3

        # Coefficient of coordinates transformation
        new_h = self.imgh * min(self.GTSize / self.imgh, self.GTSize / self.imgw)
        new_w = self.imgw * min(self.GTSize / self.imgh, self.GTSize / self.imgw)
        pad_h = (self.GTSize - new_h) / 2
        pad_w = (self.GTSize - new_w) / 2

        for idx, point_name in enumerate(self.keyPointName):
            coord = anno[idx]
            coord = coord.split(' ')
            coord = [int(coord[0]), int(coord[1])]
            GT[idx, :] = coord
            if coord[0] == -1:
                continue

            # Transformation of coordinates
            coord[0] = coord[0] * (new_w / self.imgw)
            coord[1] = coord[1] * (new_h / self.imgw)
            coord[0] += pad_w
            coord[1] += pad_h

            # Add 2d gaussian confidence map
            conf_maps[idx, :, :] = putGaussianMap(coord, conf_maps[idx, :, :], sigma, [self.GTSize, self.GTSize])
            # heatmap = plt.pcolormesh(conf_maps[idx, :, :], cmap='jet')
            # plt.show()

        return conf_maps, GT

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

        signal = cv2.resize(raw_img, (60, 64, 64))
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
                xcoord = line[2 + 2 * i]
                ycoord = line[3 + 2 * i]

                # Extract integers from string
                xcoord = int(xcoord.split('(')[1].rstrip(','))
                ycoord = int(ycoord.rstrip(')'))

                # Transfer to numpy array and assign to annotation recorder
                coord = np.array([xcoord, ycoord])
                person[self.keyPointName[i + 1]] = coord
            # Append to annotation list
            anno.append(person)

        return anno

class VGGLoader(Dataset):
    def __init__(self, dataDirectory, signal_size=60, GTSize=64, imgw=640, imgh=360, valid=0):
        '''

        '''
        # Max number of key points
        self.MAX_POINTNUM = 18

        self.dataDirectory = dataDirectory

        self.GTSize = GTSize

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
        self.signals = []
        self.GTs = []
        self.conf_maps = []

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
        for i, name in enumerate(self.names):

            # Get confidence map ground truth and signal
            conf_map, GT = self.getGroundTruth(i)
            signal = self.readSignal(name)

            self.conf_maps.append(torch.from_numpy(conf_map.astype(np.float32)))
            self.signals.append(torch.from_numpy(signal.astype(np.float32)).div(signal.max()))
            self.GTs.append(GT)




    def __getitem__(self, idx):
        '''
        Overide of __getitem__(). It acquire confidence map as ground truth
        and read raw RF signal matrix. If rotation operation is required, it
        performs rotation. The returns are confidence map and signal.
        '''
        return self.conf_maps[idx], self.signals[idx], self.GTs[idx].copy()

    def __len__(self):
        return len(self.names)
