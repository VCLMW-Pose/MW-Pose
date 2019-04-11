'''
    Created on Thu Apr 11 18:11 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   :

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import random, seed
import shutil
from src.teacher.dataset.DataLoader import *
from src.teacher.HPE import *


class Annotator():

    def __init__(self, dir):
        self.dir = dir
        self.loader = Loader(dir)
        self.estimator = HPE()

    def annotate(self):
        for i, fname, img in enumerate(self.loader):
            output = self.estimator.img_annoate(img)
            if i != 0 and i % 100 == 0:




    def save(self, output):
        with open(os.path.join(self.dir, 'annotation_all.txt'), 'a') as f:


    def distribute(self):
        with open(os.path.join(self.dir, ))


if __name__ == '__main__':
    Annotator = Annotator('/Users/midora/Desktop/MW-Pose/test')
