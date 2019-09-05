'''
    Created on Mon Mar 25 11:30 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Fri Mar 31 22:30 2019

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


class AnnotationLoader:
    """
    This class is utilized for one group of data
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
                dir: (string) Directory of folder for pre-annotated data
                        e.g. '/Users/midora/Desktop/MW-Pose/section_del/_1.0'
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
        self.data_files = list(self.annotation.keys())
        self.cur_file = 0
        self.radius = 6  # Range of discernible click
        self.drawing = False
        self.selected = ''
        self.joints = 0
        self.ix = 0
        self.iy = 0
        self.img = None  # To store th original image for clearing the skeleton.
        self.window_name = ''

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, item):
        return self.annotation[item]

    def __load_annofile(self):
        """
        Private Function
        To load pre-annotation.
        Do not care about the file architecture
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

    def lood_img(self, idx):
        if idx >= len(self):
            print("Index out of range!")
            return
        jpg = self.data_files[idx]
        if os.path.exists(os.path.join(self.dir, jpg[:-4])):
            for root, _, files in os.walk(os.path.join(self.dir, jpg[:-4])):
                for file in files:
                    if file[-4:] == '.jpg':
                        return cv2.imread(os.path.join(root, file))

    def revise(self):
        of = os.path.join(dir, 'refined.txt')
        with open(of, 'w') as f:
            for jpg in self.data_files:
                f.writelines([jpg, ' : '])
                for i, part in enumerate(self.parts):
                    if part == 'None':
                        continue
                    x = str(self.annotation[jpg][part][0])
                    y = str(self.annotation[jpg][part][1])
                    f.writelines([str(i), '(', x, ', ', y, ') '])
                f.writelines(['\n'])

    def plot_skeleton(self, img, data_file, thick):
        '''
            Args:
                window_name:    (string)
                img:            (PILImage) image for annotating
                data_file:      (string) file name e.g. 1551601527845.jpg
                thick:          (int) thick of the line
                key:            (int) the length of time the window stays
        '''
        joints = self.annotation[data_file]
        for i in range(1, len(self.parts)):
            joints[self.parts[i]] = (joints[self.parts[i]][0], joints[self.parts[i]][1])
        img = cv2.line(img, joints['rank'], joints['rkne'], (181, 102, 60), thickness=thick)
        img = cv2.line(img, joints['rkne'], joints['rhip'], (250, 203, 91), thickness=thick)
        img = cv2.line(img, joints['rhip'], joints['pelv'], (35, 98, 177), thickness=thick)
        img = cv2.line(img, joints['lhip'], joints['pelv'], (35, 98, 177), thickness=thick)
        img = cv2.line(img, joints['lhip'], joints['lkne'], (66, 218, 128), thickness=thick)
        img = cv2.line(img, joints['lkne'], joints['lank'], (62, 121, 58), thickness=thick)
        img = cv2.line(img, joints['pelv'], joints['thrx'], (23, 25, 118), thickness=thick)
        img = cv2.line(img, joints['thrx'], joints['neck'], (152, 59, 98), thickness=thick)
        img = cv2.line(img, joints['neck'], joints['head'], (244, 60, 166), thickness=thick)
        img = cv2.line(img, joints['neck'], joints['rsho'], (244, 59, 166), thickness=thick)
        img = cv2.line(img, joints['relb'], joints['rsho'], (51, 135, 239), thickness=thick)
        img = cv2.line(img, joints['rwri'], joints['relb'], (35, 98, 177), thickness=thick)
        img = cv2.line(img, joints['neck'], joints['lsho'], (244, 59, 166), thickness=thick)
        img = cv2.line(img, joints['lsho'], joints['lelb'], (49, 56, 218), thickness=thick)
        img = cv2.line(img, joints['lelb'], joints['lwri'], (23, 25, 118), thickness=thick)
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

        This function uses motion of mouse with the left bottom down to change one joint coordinate
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
            img = self.img.copy()
            self.plot_skeleton(img, self.cur_file, 2)
            cv2.imshow(self.window_name, img)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing is True:
                self.joints[self.selected] = [x, y]
            self.drawing = False
            img = self.img.copy()
            self.plot_skeleton(img, self.cur_file, 2)
            cv2.imshow(self.window_name, img)

    def MouseCallback_click(self, event, x, y, flags, param):
        """
                Standard template parameter list of mouse callback function
                Readers could refer to the following blog:
                        https://blog.csdn.net/weixin_41115751/article/details/84137783

                This function uses two click of the left mouse bottom to change one joint coordinate
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing:
                self.drawing = False
                self.annotation[self.cur_file][self.selected] = [x, y]
                self.selected = ''
                img = self.img.copy()
                self.plot_skeleton(img, self.cur_file, 2)
                cv2.imshow(self.window_name, img)
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
                    img = self.img.copy()  # To clear the original skeleton.
                    self.plot_skeleton(img, self.cur_file, 2)
                    cv2.imshow(self.window_name, img)


def refine(dir, mode):
    """
    Args:
        dir:    (string) directory of one group of data
        mode:   (string) decide whether to move or click to change the joint point
    """
    anno = AnnotationLoader(dir)
    anno.window_name = dir.split('/')[-1]
    cv2.namedWindow(anno.window_name)
    for idx, anno.cur_file in enumerate(anno.data_files):
        # if anno.data_files[idx]
        # anno.window_name = dir.split('/')[-1] + '/' + anno.cur_file
        anno.img = anno.lood_img(idx)
        # anno.img -= anno.img #  To show black canvas
        img = anno.img.copy()
        # cv2.namedWindow(anno.window_name)
        anno.plot_skeleton(img, anno.cur_file, 2)
        if mode == 'drag':
            cv2.setMouseCallback(anno.window_name, anno.MouseCallback_drag)
        elif mode == 'click':
            cv2.setMouseCallback(anno.window_name, anno.MouseCallback_click)
        else:
            print("No mode named:" + mode)
        cv2.startWindowThread()
        cv2.imshow(anno.window_name, img)
        while True:
            if cv2.waitKey(10) & 0xFF == ord('\r'):
                anno.selected = False
                anno.revise()
                break
    anno.revise()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)


def move_anno(anno_dir, dir):
    """
    Args:
        anno_dir: Top folder of pre-annotated dataset with annotation file
        dir: Top folder of dataset waiting for annotating.
    """
    if os.path.exists(anno_dir):
        for root, subdirs, _ in os.walk(anno_dir):
            for subdir in subdirs:
                src = os.path.join(root, subdir, 'joint_point.txt')
                dst = os.path.join(dir, subdir)
                if os.path.exists(src) and os.path.exists(dst):
                    shutil.copy(src, dst)

def radar_out(dir):
    anno = AnnotationLoader(dir)
    black = np.zeros((360, 640, 3))
    for idx, anno.cur_file in enumerate(anno.data_files):
        cv2.namedWindow('Black')
        anno.plot_skeleton(black, anno.cur_file, 2)
        cv2.startWindowThread()
        cv2.imshow('Black', black)
        while True:
            if cv2.waitKey(10) & 0xFF == ord('\r'):
                break

def distribute(dir):
    for root, dirs, _ in os.walk(dir):
        if root == dir:
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'joint_point.txt')):
                    os.remove(os.path.join(root, dir, 'joint_point.txt'))
    with open(os.path.join(dir, 'annotation_all.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' : ')
            name = line[0]
            point = line[1]
            name = name.split('\\')
            folder = name[0]
            jpg = name[1]
            with open(os.path.join(dir, folder, 'joint_point.txt'), 'a')as subf:
                subf.write(jpg + ' : ' + point)
    print('Distribution completed!')

if __name__ == "__main__":
    anno_dir = '/Users/midora/Desktop/MW-Pose/section_del'
    dir = 'D:/Documents/Source/MW-Pose/test/_7.0'
    # move_anno(anno_dir, dir)
    refine(dir, 'drag')
    print('Completed!')

