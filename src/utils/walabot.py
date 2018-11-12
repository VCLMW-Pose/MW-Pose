# -*- coding: utf-8 -*-
'''
    Created on Sun Nov 11 10:46 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sun Nov 11 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import matplotlib.pyplot as plt
import WalabotAPI
import numpy as np
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

    def scan(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=True):
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
        self.fig = plt.figure(figsize=((maxPhi - minPhi), (maxTheta - minTheta)))
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_xlim(minPhi, maxPhi)
        #self.ax.set_ylim(minTheta, maxTheta)

        M, _, _, _, _ = self.walabot.GetRawImage()
        M = np.array(M)[5]
        self.heatmap = self.ax.pcolormesh(M, cmap='jet')
        self.fig.colorbar(self.heatmap)

        anima = animation.FuncAnimation(self.fig, self.update, init_func=self.init, repeat=False, interval=0,
                                                                                                        blit=True)
        plt.show()

    def update(self, image):
        '''
        update routine for animation
        '''
        self.walabot.Trigger()
        rawimage, _, _, _, _ = self.walabot.GetRawImage()
        rawimage = np.array(rawimage)[5]
        self.heatmap = self.ax.pcolormesh(rawimage, cmap='jet')
        return self.heatmap,

if __name__ == '__main__':
    Walabot = walabot()
    Walabot.scan(10, 300, 10, -30, 30, 2, -30 ,30 ,2, 35)






