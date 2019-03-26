'''
    Created on Mon Mar 25 11:30 2019

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


class Annotation:
    """
    Structure of annotation:
        dict{
            key: file name (e.g. '1551601527845.jpg')
            value: dictionary of joints{
                                        key: joint name (e.g. 'head')
                                        value: coordinate[x, y] (e.g. [125, 320])
                                        }
            }
    """
    def __init__(self, dir):
        """
            Args:
                dir: Directory of folder for pre-annotated data
                        e.g. /Users/midora/Desktop/MW-Pose/section_del/_1.0
        """
        self.dir = dir
        if os.path.exists(os.path.join(dir, 'refined.txt')):
            self.anno_file = os.path.join(dir, 'refined.txt')
        else:
            self.anno_file = os.path.join(dir, 'joint_point.txt')
        self.annotation = {}
        self.parts = ['None',  # To be compatible with the pre-annotation
                      'rank', 'rkne', 'rhip',
                      'lhip', 'lkne', 'lank',
                      'pelv', 'thrx', 'neck', 'head',
                      'rwri', 'relb', 'rsho',
                      'lsho', 'lelb', 'lwri']
        """
        rank: right ankle
        rkne: right knee
        rhip: right hip
        lhip: left hip
        lkne: left knee
        lank: left ankle
        pelv: pelvis
        thrx: thorax
        neck: neck
        head: head
        rwri: right wrist
        relb: right elbow
        rsho: right shoulder
        lsho: left shoulder
        lelb: left elbow
        lwri: left wrist
        """
        self.__load_annofile()
        self.data_files = self.annotation.keys()
        self.cur_file = 0
        self.radius = 6  # Range of discernible click
        self.drawing = False
        self.selected = ''
        self.joints = 0
        self.ix = 0
        self.iy = 0
        self.img = None  # To store th original image for clearing the skeleton.

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, item):
        return self.annotation[item]

    def __load_annofile(self):
        """
        Private Function
        To load pre-annotation.
        """
        with open(self.anno_file) as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' : ')
            name = line[0]
            joints = line[1].rstrip(') \n').split(') ')
            dict_joints = {}
            for joint in joints:
                joint = joint.split('(')
                str_coor = joint[1].split(', ')
                dict_joints[self.parts[int(joint[0])]] = [int(str_coor[0]), int(str_coor[1])]
            self.annotation[name] = dict_joints

    def __revise(self):
        with open(dir)


    def plot_skeleton(self, img, data_file, thick):
        '''
            Args:
                window_name: (string)
                img: (PILImage) image for annotating
                data_file: (string) file name e.g. 1551601527845.jpg
                thick: (int) thick of the line
                key: (int) the length of time the window stays
        '''
        joints = self.annotation[data_file]
        for i in range(1, len(self.parts)):
            joints[self.parts[i]] = (joints[self.parts[i]][0], joints[self.parts[i]][1])
        img = cv2.line(img, joints['rank'], joints['rkne'], (181, 102, 60), thickness=thick)
        img = cv2.line(img, joints['rkne'], joints['rhip'], (250, 203, 91), thickness=thick)
        img = cv2.line(img, joints['rhip'], joints['pelv'], (35, 98, 177),  thickness=thick)
        img = cv2.line(img, joints['lhip'], joints['pelv'], (35, 98, 177),  thickness=thick)
        img = cv2.line(img, joints['lhip'], joints['lkne'], (66, 218, 128), thickness=thick)
        img = cv2.line(img, joints['lkne'], joints['lank'], (62, 121, 58),  thickness=thick)
        img = cv2.line(img, joints['pelv'], joints['thrx'], (23, 25, 118),  thickness=thick)
        img = cv2.line(img, joints['thrx'], joints['neck'], (152, 59, 98),  thickness=thick)
        img = cv2.line(img, joints['neck'], joints['head'], (244, 60, 166), thickness=thick)
        img = cv2.line(img, joints['neck'], joints['rsho'], (244, 59, 166), thickness=thick)
        img = cv2.line(img, joints['relb'], joints['rsho'], (51, 135, 239), thickness=thick)
        img = cv2.line(img, joints['rwri'], joints['relb'], (35, 98, 177),  thickness=thick)
        img = cv2.line(img, joints['neck'], joints['lsho'], (244, 59, 166), thickness=thick)
        img = cv2.line(img, joints['lsho'], joints['lelb'], (49, 56, 218),  thickness=thick)
        img = cv2.line(img, joints['lelb'], joints['lwri'], (23, 25, 118),  thickness=thick)
        for joint in self.parts:
            if joint == 'None':
                continue
            if joint == self.selected:
                # Highlight the selected joint.
                img = cv2.circle(img, joints[joint], 5, (0, 255, 0), -1)
            else:
                img = cv2.circle(img, joints[joint], 3, (68, 147, 200), -1)
        return img

    def MouseCallback_drag(self, event, x, y, flags, param):
        """
        Standard template parameter list of mouse callback function
        Readers could refer to the following blog:
                https://blog.csdn.net/weixin_41115751/article/details/84137783
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            ix, iy = x, y
            self.joints = self.annotation[self.cur_file]
            self.selected = ''
            min_radius = self.radius
            for joint in self.parts:
                if joint == 'None':
                    continue
                if abs(self.joints[joint][0] - ix) + abs(self.joints[joint][1] - iy) < min_radius:
                    self.selected = joint
                    min_radius = abs(self.joints[joint][0] - ix) + abs(self.joints[joint][1] - iy)

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if self.drawing is True:
                self.joints[self.selected] = [x, y]
            img = anno.img.copy()
            self.plot_skeleton(img, self.cur_file, 2)
            cv2.imshow(window_name, img)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing is True:
                self.joints[self.selected] = [x, y]
            self.drawing = False
            img = anno.img.copy()
            self.plot_skeleton(img, self.cur_file, 2)
            cv2.imshow(window_name, img)

    def MouseCallback_click(self, event, x, y, flags, param):
        """
                Standard template parameter list of mouse callback function
                Readers could refer to the following blog:
                        https://blog.csdn.net/weixin_41115751/article/details/84137783
                """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing:
                self.drawing = False
                self.annotation[self.cur_file][self.selected] = [x, y]
                self.selected = ''
                img = anno.img.copy()
                self.plot_skeleton(img, self.cur_file, 2)
                cv2.imshow(window_name, img)
            else:
                self.joints = self.annotation[self.cur_file]
                self.selected = ''
                min_radius = self.radius
                for joint in self.parts:
                    if joint == 'None':
                        continue
                    if abs(self.joints[joint][0] - x) + abs(self.joints[joint][1] - y) < min_radius:
                        # Using 1-norm to replace the Euclid distance to reduce the quantity of calculation.
                        self.selected = joint
                        min_radius = abs(self.joints[joint][0] - x) + abs(self.joints[joint][1] - y)
                        # The joint selected should be with the least "distance".
                        self.drawing = True
                if self.drawing:
                    img = anno.img.copy()  # To clear the original skeleton.
                    self.plot_skeleton(img, self.cur_file, 2)
                    cv2.imshow(window_name, img)


def load_img(dir, file):
    img = cv2.imread(os.path.join(dir, file))
    cv2.namedWindow(dir.split('/')[-1] + '/' + file)
    return img


def annotate(dir):
    if os.path.exists(os.path.join(dir, 'joint_point.txt')):
        with open(os.path.join(dir, 'joint_point.txt'), 'r') as f:
            lines = f.reandlines()
    for _, dirs, _ in os.walk(dir, topdown=True):
        pass


if __name__ == "__main__":
    dir = '/Users/midora/Desktop/MW-Pose/section_del/_7.0'
    anno = Annotation(dir)
    for anno.cur_file in anno.data_files:
        window_name = dir.split('/')[-1] + '/' + anno.cur_file
        anno.img = cv2.imread(os.path.join(dir, anno.cur_file))
        img = anno.img.copy()
        cv2.namedWindow(window_name)
        anno.plot_skeleton(img, anno.cur_file, 2)
        cv2.setMouseCallback(window_name, anno.MouseCallback_drag)
        cv2.putText(img, str(random()), (100, 100), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=2)
        cv2.imshow(window_name, img)
        img = anno.img.copy()
        while (True):
            try:
                cv2.waitKey(100)
            except Exception:
                cv2.destroyAllWindows()
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    cv2.waitKey(0)
    # while(cv2.waitKey(10) != 'q'):
    #     continue
    print('Completed!')
