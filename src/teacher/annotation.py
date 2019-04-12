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
        for i, (fname, img) in enumerate(self.loader):
            output = self.estimator.img_annotate(img)
            self.save(fname, output)
            print('%d image have been annotated.' % i)



    def save(self, fname, output):
        fname = fname.split('\\')
        jpg = fname[-3] + '\\' + fname[-1]
        with open(os.path.join(self.dir, 'annotation_all.txt'), 'a') as f:
                f.writelines([jpg, ' : '])
                for i in range(0, 16):
                    x = str(output[i][0])
                    y = str(output[i][1])
                    f.writelines([str(i+1), '(', x, ', ', y, ') '])
                f.writelines(['\n'])


    def distribute(self):
        for root, dirs, _ in os.walk(self.dir):
            if root == self.dir:
                for dir in dirs:
                    if os.path.exists(os.path.join(root, dir, 'joint_point.txt')):
                        os.remove(os.path.join(root, dir, 'joint_point.txt'))
        with open(os.path.join(self.dir, 'annotation_all.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' : ')
                name = line[0]
                point = line[1]
                name = name.split('\\')
                folder = name[0]
                jpg = name[1]
                with open(os.path.join(self.dir, folder, 'joint_point.txt'), 'a')as subf:
                    subf.write(jpg + ' : ' + point)
        print('Distribution completed!')



if __name__ == '__main__':
    annotator = Annotator('D:\\Documents\\Source\\MW-Pose\\test')
    # annotator.annotate()
    annotator.distribute()