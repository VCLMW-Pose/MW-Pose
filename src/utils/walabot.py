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
        self.walabot = WalabotAPI
        self.walabot.Init()
        self.walabot.SetSettingsFolder()
        print('Walabot initialized!')

    def __delete__(self):
        self.walabot.Stop()
        self.walabot.Disconnect()

    def init(self):
        return self.heatmap,

    def scan(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=True):
        self.walabot.ConnectAny()
        self.walabot.SetProfile(self.walabot.PROF_SENSOR)
        self.walabot.SetArenaR(minR, maxR, resR)
        self.walabot.SetArenaTheta(minTheta, maxTheta, resTheta)
        self.walabot.SetArenaPhi(minPhi, maxPhi, resPhi)
        self.walabot.SetThreshold(threshold)
        self.phi, self.theta = list(range(minPhi, maxPhi, resPhi)) + [maxPhi], \
                                                            list(range(minTheta, maxTheta, resTheta)) + [maxTheta]
        self.pos = np.array([list((phi, theta)) for phi in self.phi for theta in self.theta]).transpose()

        if mti:
            self.walabot.SetDynamicImageFilter(self.walabot.FILTER_TYPE_MTI)
        print('Walabot configuration complete!')
        self.walabot.Start()
        self.walabot.StartCalibration()
        print('Walabot proceeding scanning!')

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
        self.walabot.Trigger()
        rawimage, _, _, _, _ = self.walabot.GetRawImage()
        rawimage = np.array(rawimage)[5]
        self.heatmap = self.ax.pcolormesh(rawimage, cmap='jet')
        return self.heatmap,

if __name__ == '__main__':
    Walabot = walabot()
    Walabot.scan(10, 300, 10, -30, 30, 2, -30 ,30 ,2, 35)






