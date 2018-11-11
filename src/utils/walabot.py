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

    def __delete__(self):
        self.walabot.Stop()
        self.walabot.Disconnect()

    def Initialize(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=True):
        self.walabot.ConnectAny()
        self.walabot.SetProfile(self.walabot.PROF_SENSOR)
        self.walabot.SetArenaR(minR, maxR, resR)
        self.walabot.SetArenaTheta(minTheta, maxTheta, resTheta)
        self.walabot.SetArenaPhi(minPhi, maxPhi, resPhi)
        self.walabot.SetThreshold(threshold)
        self.phi, self.theta = np.meshgrid(np.linespace(minPhi, maxPhi, (maxPhi - minPhi)/2 + 1),
                                           np.linespace(minTheta, maxTheta, (maxTheta, minTheta)/2 + 1))

        if mti:
            self.walabot.SetDynamicImageFilter(self.walabot.FILTER_TYPE_MTI)

        self.walabot.Start()
        self.walabot.StartCalibration()
        plt.figure()

        def update():
            self.walabot.Trigger()
            rawimage, _, _, _, _ = self.walabot.GetRawImage()
            rawimage = np.array(rawimage)

            print(rawimage.shape)

    def get_image(self):
        active = True
        while active:
            self.walabot.Trigger()
            rawimage, _, _, _, _ = self.walabot.GetRawImage()
            rawimage = np.array(rawimage)
            return rawimage

if __name__ == '__main__':
    Walabot = walabot()
    Walabot.Initialize(10, 300, 10, -30, 30, 2, -30 ,30 ,2, 15)






