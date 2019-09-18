'''
    Created on Thu Apr 11 18:11 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Thu Sep 19 00:05 2019

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

__all__ = ['Saver']

class Saver():

    def __init__(self, dir):
        self.dir = dir
        # self.loader = Loader(dir)
        if os.path.exists(os.path.join(self.dir, 'annotation_all.txt')):
            os.remove(os.path.join(self.dir, 'annotation_all.txt'))

    def crawl(self, fname, joint_list, person_to_joint_assoc):
        """
        Parameter:
         joint_list:  (n x 5 ndarray) n = number of joint in the image
                                row: one joint of someone
                                column 0: X
                                column 1: Y
                                column 3: sequence number(row number)
                                column 4: type (which joint)
                                    Value check list:
                                    [0]: nose
                                    [1]: neck
                                    [2]: rShoulder
                                    [3]: rElbow
                                    [4]: rWrist
                                    [5]: lShoulder
                                    [6]: lElbow
                                    [7]: lWrist
                                    [8]: rHip
                                    [9]: rKnee
                                    [10]: rAnkle
                                    [11]: lHip
                                    [12]: lKnee
                                    [13]: lAnkle
                                    [14]: rEye
                                    [15]: lEye
                                    [16]: rEar
                                    [17]: lEar
         person_to_joint_assoc: (n x 20 ndarray) n = number of people in the image
                                row: one person
                                column: joint type number
                                value: sequence number in joint_list(-1 means not including)

        """
        for person_num, person in enumerate(person_to_joint_assoc):
            output = []
            for joint_t in range(0, 18):
                row = int(person[joint_t])
                if row == -1:
                    output.append([-1, -1, -1])
                else:
                    output.append([joint_list[row][0], joint_list[row][1], joint_list[row][2]])
            self.save(fname, person_num, output)




    def save(self, fname, person_num, output):

        #Parameter:
        #    fname: e.g. _12.0\xxxxxxxxx.jpg
        #    person_num: person number in the image
        #    output: (18 x 2 ndarray) joint coordinate

        fname = fname.split('\\')
        jpg = fname[-3] + '\\' + fname[-1]
        with open(os.path.join(self.dir, 'annotation_all.txt'), 'a') as f:
                f.writelines([jpg, ' : ', str(person_num), ' '])
                for i in range(0, 18):
                    x = str(output[i][0])
                    y = str(output[i][1])
                    c = str(output[i][2])
                    f.writelines([str(i+1), '(', x, ', ', y, ', ', c, ') '])
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
    annotator = Saver('D:\\Documents\\Source\\MW-Pose\\test')
