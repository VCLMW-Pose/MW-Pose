# -*- coding: utf-8 -*-
'''
    Created on Sun Nov 11 10:46 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Mon Nov 21 16:15 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

__all__ = ['walabot']

import matplotlib.pyplot as plt
import WalabotAPI
import numpy as np
import os
from src.utils.imageproc import *
from PIL import Image
from sys import platform
from matplotlib import animation

class walabot():
    def __init__(self):
        '''
        Walabot initialization
        '''
        self.walabot = WalabotAPI
        self.walabot.Init()
        self.walabot.SetSettingsFolder()

    def __delete__(self):
        self.walabot.Stop()
        self.walabot.Disconnect()

    def init(self):
        '''
        init routine for animation
        '''
        return self.heatmap,

    def scan_test(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mode, mti=True):
        '''
        Args:
             minR        : (int) scan arena configuration parameter, minimum distance
             maxR        : (int) maximum distance of scan arena
             resR        : (float) resolution of depth
             minTheta    : (int) minimum theta
             maxTheta    : (int) maximum theta
             resTheta    : (int) vertical angular resolution
             minPhi      : (int) minimum phi
             maxPhi      : (int) maximum phi
             resPhi      : (int) horizontal angular resolution
             threshold   : (int) threshold for weak signals
             mode        : (string) scan mode
             mti         : (boolean) ignore static reflectors
        '''
        # Walabot configuration
        self.walabot.ConnectAny()
        self.walabot.SetProfile(self.walabot.PROF_SENSOR)
        self.walabot.SetArenaR(minR, maxR, resR)
        self.walabot.SetArenaTheta(minTheta, maxTheta, resTheta)
        self.walabot.SetArenaPhi(minPhi, maxPhi, resPhi)
        self.walabot.SetThreshold(threshold)

        # Ignore static reflector
        if mti:
            self.walabot.SetDynamicImageFilter(self.walabot.FILTER_TYPE_MTI)

        # Start scanning
        self.walabot.Start()
        self.walabot.StartCalibration()

        # Plot animation
        self.fig = plt.figure()
        #self.fig = plt.figure(figsize=((maxPhi - minPhi), (maxTheta - minTheta)))
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_xlim(minPhi, maxPhi)
        #self.ax.set_ylim(minTheta, maxTheta)

        if mode == "horizontal":
            M, _, _, _, _ = self.walabot.GetRawImageSlice()
        elif mode == "perpendicular":
            M, _, _, _, _ = self.walabot.GetRawImage()
            M = sumup_perpendicular(np.array(M))
            M.transpose(1, 0)

        self.heatmap = self.ax.pcolormesh(M, cmap='jet')
        self.fig.colorbar(self.heatmap)

        if mode == "horizontal":
            anima = animation.FuncAnimation(self.fig, self.updateslice, init_func=self.init, repeat=False, interval=0,
                                                                                                        blit=True)
        elif mode == "perpendicular":
            anima = animation.FuncAnimation(self.fig, self.update, init_func=self.init, repeat=False, interval=0,
                                                                                                        blit=True)
        plt.show()

    def update(self, image):
        '''
        update routine for animation
        '''
        self.walabot.Trigger()
        rawimage, _, _, _, _ = self.walabot.GetRawImage()
        rawimage = sumup_perpendicular(np.array(rawimage))
        rawimge = rawimage.transpose(1, 0)
        self.heatmap = self.ax.pcolormesh(rawimage, cmap='jet')
        return self.heatmap,

    def updateslice(self, image):
        '''
        update routine for animation
        '''
        self.walabot.Trigger()
        rawimage, _, _, _, _ = self.walabot.GetRawImageSlice()
        rawimage = np.array(rawimage)
        self.heatmap = self.ax.pcolormesh(rawimage, cmap='jet')
        return self.heatmap,

    def initialize(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=True):
        '''
        Args:
             minR        : (int) scan arena configuration parameter, minimum distance
             maxR        : (int) maximum distance of scan arena
             resR        : (float) resolution of depth
             minTheta    : (int) minimum theta
             maxTheta    : (int) maximum theta
             resTheta    : (int) vertical angular resolution
             minPhi      : (int) minimum phi
             maxPhi      : (int) maximum phi
             resPhi      : (int) horizontal angular resolution
             threshold   : (int) threshold for weak signals
             mti         : (boolean) ignore static reflectors
        Returns:
             Initialize walabot and complete configuration
        '''
        # Walabot configuration
        self.walabot.ConnectAny()
        self.walabot.SetProfile(self.walabot.PROF_SENSOR)
        self.walabot.SetArenaR(minR, maxR, resR)
        self.walabot.SetArenaTheta(minTheta, maxTheta, resTheta)
        self.walabot.SetArenaPhi(minPhi, maxPhi, resPhi)
        self.walabot.SetThreshold(threshold)

        # Ignore static reflector
        if mti:
            self.walabot.SetDynamicImageFilter(self.walabot.FILTER_TYPE_MTI)

        # Start scanning
        self.walabot.Start()
        self.walabot.StartCalibration()

    def get_frame(self):
        self.walabot.Trigger()
        # Getting heat maps and add up to a 2D matrix
        heatmap, _, _, _, _ = self.walabot.GetRawImage()
        heatmap = np.array(heatmap)
        heatmap = sumup(heatmap)

        # Generalization
        heatmap = (heatmap + heatmap.min())/heatmap.max()

        return heatmap

    def get_frame_slice(self):
        self.walabot.Trigger()
        # Getting heat maps in R and phi
        heatmap, _, _, _, _ = self.walabot.GetRawImageSlice()
        heatmap = np.array(heatmap)

        return heatmap

    def get_frame_verticle(self):
        self.walabot.Trigger()
        # Getting heat maps in R and theta
        heatmap, _, _, _, _ = self.walabot.GetRawImage()

def read_signal_test(save_dir):
    names = os.listdir(save_dir)

    for i in range(len(names)):
        sig_file = open(os.path.join(save_dir, names[i]), mode='rb')
        data = np.fromfile(sig_file, dtype=np.int32)

        size_x = data[0]
        size_y = data[1]
        size_z = data[2]

        raw_img = np.array(data[3:]).reshape(size_x, size_y, size_z)
        _perpendicular = sumup_perpendicular(np.copy(raw_img))
        _horizontal = sumup_horizontal(np.copy(raw_img))

        figure = plt.figure()

def read_signal(save_dir):
    sig_file = open(save_dir, mode='rb')
    data = np.fromfile(sig_file, dtype=np.int32)

    size_x = data[0]
    size_y = data[1]
    size_z = data[2]

    raw_img = np.array(data[3:]).reshape(size_z, size_x, size_y)
    _horizontal = sumup_horizontal(np.copy(raw_img))
    heatmap = plt.pcolormesh(_horizontal, cmap='jet')
    plt.show()
    # plt.savefig(save_dir + "horizontal.jpg")


if __name__ == '__main__':
    read_signal("F:/capref2/signals/1551601908119")
    # read_signal("E:/MW-Pose/res/1551761233560")