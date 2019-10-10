'''
    Created on Thu Sep 5 22:46 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Mon Sep 23 00:13 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

'''
*******************************************************************************
Description:
    This file is based on refine.py and revised to be compatible with OpenPose.
*******************************************************************************
'''

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import random, seed
import shutil
import tkinter as tk
from PIL import Image, ImageTk

# Key value of Backspace
mac = 127
win = 8

add_joint_num = ''


class AnnotationLoader:
    """
    This class is utilized for one group of data
    Architecture of annotation:
        dict{
            key: file name (e.g. '1551601527845.jpg')
            value: list of people[
                Every people has a architecture below:
                    dictionary of joints{
                                        key: joint name (e.g. 'head')
                                        value: coordinate[x, y] (e.g. [125, 320])
                                        }
                                ]
            }
    """

    def __init__(self, dir, thread=0):
        """
            Args:
                dir: (string) Directory of folder for pre-annotated data
                        e.g. '/Users/midora/Desktop/MW-Pose/section_del/_1.0'
        """
        self.dir = dir
        self.thread = thread
        if os.path.exists(os.path.join(dir, 'refined.txt')):
            self.anno_file = os.path.join(dir, 'refined.txt')
        else:
            self.anno_file = os.path.join(dir, 'joint_point.txt')
        self.annotation = {}
        self.parts = ['None',  # To be compatible with the pre-annotation
                      'nose', 'neck', 'rShoulder',
                      'rElbow', 'rWrist', 'lShoulder',
                      'lElbow', 'lWrist', 'rHip', 'rKnee',
                      'rAnkle', 'lHip', 'lKnee', 'lAnkle',
                      'rEye', 'lEye', 'rEar', 'lEar']
        """
        Value check list:(Serial number should be plussed one.)
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
        """
        self.__load_annofile()
        self.data_files = list(self.annotation.keys())
        self.cur_file = 0  # file name(e.g. xxxxxxxx.jpg)
        self.radius = 6  # Range of discernible click
        self.drawing = False
        self.deleting = False
        self.adding = False
        self.add_joint_num = -1
        self.person_selected = -1
        self.joint_selected = ''
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
        with open(self.anno_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' : ')
            name = line[0]
            if name not in self.annotation:
                self.annotation[name] = []
            line = line[1][2:]
            dict_joints = {}
            joints = line.rstrip(') \n').split(') ')
            confidence = 0
            for joint in joints:
                joint = joint.split('(')
                str_coor = joint[1].split(', ')
                dict_joints[self.parts[int(joint[0])]] = [int(float(str_coor[0])), int(float(str_coor[1])),
                                                          float(str_coor[2])]
                confidence += float(str_coor[2])
            if confidence > self.thread:
                self.annotation[name].append(dict_joints)

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
                for i, people in enumerate(self.annotation[jpg]):
                    f.writelines([jpg, ' : ', str(i), ' '])
                    for j, part in enumerate(self.parts):
                        if part == 'None':
                            continue
                        x = str(people[part][0])
                        y = str(people[part][1])
                        c = str(people[part][2])
                        f.writelines([str(j), '(', x, ', ', y, ', ', c, ') '])
                    f.writelines(['\n'])
        self.drawing = False
        self.deleting = False
        self.person_selected = -1
        self.joint_selected = ''

    def plot_skeleton(self, img, data_file, thick):
        '''
            Args:
                window_name:    (string)
                img:            (PILImage) image for annotating
                data_file:      (string) file name e.g. 1551601527845.jpg
                thick:          (int) thick of the line
                key:            (int) the length of time the window stays
        '''
        for people, joints in enumerate(self.annotation[data_file]):
            jointscoor = {}
            for i in range(1, len(self.parts)):
                jointscoor[self.parts[i]] = (joints[self.parts[i]][0], joints[self.parts[i]][1])
            if jointscoor['nose'][0] != -1 and jointscoor['neck'][0] != -1:
                img = cv2.line(img, jointscoor['nose'], jointscoor['neck'], (181, 102, 60), thickness=thick)
            if jointscoor['neck'][0] != -1 and jointscoor['rShoulder'][0] != -1:
                img = cv2.line(img, jointscoor['neck'], jointscoor['rShoulder'], (250, 203, 91), thickness=thick)
            if jointscoor['rShoulder'][0] != -1 and jointscoor['rElbow'][0] != -1:
                img = cv2.line(img, jointscoor['rShoulder'], jointscoor['rElbow'], (35, 198, 77), thickness=thick)
            if jointscoor['rElbow'][0] != -1 and jointscoor['rWrist'][0] != -1:
                img = cv2.line(img, jointscoor['rElbow'], jointscoor['rWrist'], (35, 98, 177), thickness=thick)
            if jointscoor['neck'][0] != -1 and jointscoor['lShoulder'][0] != -1:
                img = cv2.line(img, jointscoor['neck'], jointscoor['lShoulder'], (66, 218, 128), thickness=thick)
            if jointscoor['lShoulder'][0] != -1 and jointscoor['lElbow'][0] != -1:
                img = cv2.line(img, jointscoor['lShoulder'], jointscoor['lElbow'], (62, 121, 58), thickness=thick)
            if jointscoor['lElbow'][0] != -1 and jointscoor['lWrist'][0] != -1:
                img = cv2.line(img, jointscoor['lElbow'], jointscoor['lWrist'], (23, 25, 118), thickness=thick)
            if jointscoor['neck'][0] != -1 and jointscoor['rHip'][0] != -1:
                img = cv2.line(img, jointscoor['neck'], jointscoor['rHip'], (152, 59, 98), thickness=thick)
            if jointscoor['rHip'][0] != -1 and jointscoor['rKnee'][0] != -1:
                img = cv2.line(img, jointscoor['rHip'], jointscoor['rKnee'], (94, 160, 66), thickness=thick)
            if jointscoor['rKnee'][0] != -1 and jointscoor['rAnkle'][0] != -1:
                img = cv2.line(img, jointscoor['rKnee'], jointscoor['rAnkle'], (44, 159, 96), thickness=thick)
            if jointscoor['neck'][0] != -1 and jointscoor['lHip'][0] != -1:
                img = cv2.line(img, jointscoor['neck'], jointscoor['lHip'], (51, 135, 239), thickness=thick)
            if jointscoor['lHip'][0] != -1 and jointscoor['lKnee'][0] != -1:
                img = cv2.line(img, jointscoor['lHip'], jointscoor['lKnee'], (75, 58, 217), thickness=thick)
            if jointscoor['lKnee'][0] != -1 and jointscoor['lAnkle'][0] != -1:
                img = cv2.line(img, jointscoor['lKnee'], jointscoor['lAnkle'], (244, 59, 166), thickness=thick)
            if jointscoor['nose'][0] != -1 and jointscoor['rEye'][0] != -1:
                img = cv2.line(img, jointscoor['nose'], jointscoor['rEye'], (49, 56, 218), thickness=thick)
            if jointscoor['rEye'][0] != -1 and jointscoor['rEar'][0] != -1:
                img = cv2.line(img, jointscoor['rEye'], jointscoor['rEar'], (23, 25, 118), thickness=thick)
            if jointscoor['nose'][0] != -1 and jointscoor['lEye'][0] != -1:
                img = cv2.line(img, jointscoor['nose'], jointscoor['lEye'], (130, 35, 158), thickness=thick)
            if jointscoor['lEye'][0] != -1 and jointscoor['lEar'][0] != -1:
                img = cv2.line(img, jointscoor['lEye'], jointscoor['lEar'], (53, 200, 18), thickness=thick)
            for joint in self.parts:
                if joint == 'None':
                    continue
                if self.deleting and people == self.person_selected:
                    img = cv2.circle(img, jointscoor[joint], 5, (0, 0, 255), -1)
                elif joint == self.joint_selected and people == self.person_selected:
                    # Highlight the selected joint.
                    img = cv2.circle(img, jointscoor[joint], 5, (0, 255, 0), -1)
                elif jointscoor[joint] != (-1, -1):
                    img = cv2.circle(img, jointscoor[joint], 3, (68, 147, 200), -1)
        return img

    def __find_joint(self, x, y, ifdel=False):
        '''
        Call this function to figure out which joint the user want to click.
        The result will be store in self.person_selected, self.joint_selected and self.joints
        Parameter:
            x: Mouse X position
            y: Mouse Y position
        '''
        ix, iy = x, y
        self.person_selected = -1
        self.joint_selected = ''
        self.drawing = False
        self.deleting = False
        for person, joints in enumerate(self.annotation[self.cur_file]):  # Every cycle dispose one person
            min_radius = self.radius
            for joint in self.parts:
                if joint == 'None':
                    continue
                if abs(joints[joint][0] - ix) + abs(joints[joint][1] - iy) < min_radius:
                    if ifdel:
                        self.deleting = True
                    else:
                        self.drawing = True
                    self.person_selected = person
                    self.joint_selected = joint
                    self.joints = joints
                    min_radius = abs(self.joints[joint][0] - ix) + abs(self.joints[joint][1] - iy)
        img = self.img.copy()  # clear all skeleton drawn on the image
        self.plot_skeleton(img, self.cur_file, thick=2)
        cv2.imshow(self.window_name, img)

    def MouseCallback_drag(self, event, x, y, flags, param):
        """
        Standard template parameter list of mouse callback function
        Readers could refer to the following blog:
                https://blog.csdn.net/weixin_41115751/article/details/84137783

        This function uses motion of mouse with the left bottom down to change one joint coordinate
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.adding:
                self.annotation[self.cur_file][0][self.parts[self.add_joint_num]] = [x, y, 1]
                self.adding = False
                self.person_selected = 0
                self.joint_selected = self.parts[self.add_joint_num]
                img = self.img.copy()  # clear all skeleton drawn on the image
                self.plot_skeleton(img, self.cur_file, thick=2)
                cv2.imshow(self.window_name, img)
            else:
                self.__find_joint(x, y)

        elif (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON) or event == cv2.EVENT_LBUTTONUP:
            self.deleting = False
            if self.drawing:
                if x <= 0 or y <= 0:
                    self.joints[self.joint_selected] = [-1, -1, -1]
                else:
                    self.joints[self.joint_selected] = [x, y, 1]
                #  self.joints here is actually a pointer pointed some part of self.annotation
                #  So only changing self.joints is OK.
            # Refresh image
            img = self.img.copy()  # clear all skeleton drawn on the image
            self.plot_skeleton(img, self.cur_file, thick=2)
            cv2.imshow(self.window_name, img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.__find_joint(x, y, ifdel=True)

        # elif event == cv2.EVENT_RBUTTONDBLCLK and flags == cv2.EVENT_FLAG_CTRLKEY:
        #     if self.deleting:
        #         del self.annotation[self.cur_file][self.person_selected]
        #         self.deleting = False
        #         self.person_selected = -1
        #         self.joint_selected = ''
        #         img = self.img.copy()  # clear all skeleton drawn on the image
        #         self.plot_skeleton(img, self.cur_file, thick=2)
        #         cv2.imshow(self.window_name, img)

    ##########  This function has not been compatible with OpenPose annotation yet. ####################
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
                self.annotation[self.cur_file][self.joint_selected] = [x, y]
                self.joint_selected = ''
                img = self.img.copy()
                self.plot_skeleton(img, self.cur_file, 2)
                cv2.imshow(self.window_name, img)
            else:
                self.joints = self.annotation[self.cur_file]
                self.joint_selected = ''
                min_radius = self.radius
                for joint in self.parts:
                    if joint == 'None':
                        continue
                    if abs(self.joints[joint][0] - x) + abs(self.joints[joint][1] - y) < min_radius:
                        # Using 1-norm to replace the Euclid distance to reduce the quantity of calculation.
                        self.joint_selected = joint
                        min_radius = abs(self.joints[joint][0] - x) + abs(self.joints[joint][1] - y)
                        # The joint selected should be with the least "distance".
                        self.drawing = True
                if self.drawing:
                    img = self.img.copy()  # To clear the original skeleton.
                    self.plot_skeleton(img, self.cur_file, 2)
                    cv2.imshow(self.window_name, img)


class MyCollectApp(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title('Add missing joint')
        self.setupUI()

    def setupUI(self):
        row1 = tk.Frame(self)
        row1.pack(fill="x")
        l1 = tk.Label(row1, text="Joint Serial Number：", height=10, width=20)
        l1.pack(side=tk.LEFT)  # 这里的side可以赋值为LEFT  RTGHT TOP  BOTTOM
        self.xls_text = tk.StringVar()
        tk.Entry(row1, textvariable=self.xls_text).pack(side=tk.RIGHT)

        row2 = tk.Frame(self)
        row2.pack(fill="x")
        tk.Button(row2, text="Confirm", command=self.on_click).pack(side=tk.RIGHT)

    def on_click(self):
        global add_joint_num
        add_joint_num = self.xls_text.get().lstrip()
        if len(add_joint_num) == 0:
            messagebox.showwarning(title='Warning', message='Please enter the joint number!')
            return False
        self.quit()
        self.destroy()
        print("You added joint：%s" % (add_joint_num))

def pop_box():
    window = tk.Tk()
    window.title('Joint Addition')
    window.geometry('500x300')  # 这里的乘是小x

    def print_selection():
        add_joint_num = lb.get(lb.curselection())  # 获取当前选中的文本
        window.destroy()
        print(add_joint_num)

    b1 = tk.Button(window, text='Confirm', width=15, height=2, command=print_selection)
    b1.pack()
    var2 = tk.StringVar()
    var2.set(['nose', 'neck', 'rShoulder',
              'rElbow', 'rWrist', 'lShoulder',
              'lElbow', 'lWrist', 'rHip', 'rKnee',
              'rAnkle', 'lHip', 'lKnee', 'lAnkle',
              'rEye', 'lEye', 'rEar', 'lEar'])  # 为变量var2设置值
    lb = tk.Listbox(window, listvariable=var2)  # 将var2的值赋给Listbox
    lb.pack()

    window.mainloop()

def refine(dir, mode, thread=0, os=mac):
    """
    Args:
        dir:    (string) directory of one group of data
        mode:   (string) decide whether to move or click to change the joint point
    """
    anno = AnnotationLoader(dir, thread)
    # app = MyCollectApp()
    anno.window_name = dir.split('/')[-1]
    cv2.namedWindow(anno.window_name)
    idx = 0
    while idx < len(anno.data_files):
        anno.cur_file = anno.data_files[idx]
        # if anno.data_files[idx]
        # anno.window_name = dir.split('/')[-1] + '/' + anno.cur_file
        anno.img = anno.lood_img(idx)
        # anno.img -= anno.img #  To show black canvas
        img = anno.img.copy()
        # img = np.zeros((360, 640, 3))
        # cv2.namedWindow(anno.window_name)
        anno.plot_skeleton(img, anno.cur_file, thick=2)
        if mode == 'drag':
            cv2.setMouseCallback(anno.window_name, anno.MouseCallback_drag)
        elif mode == 'click':
            cv2.setMouseCallback(anno.window_name, anno.MouseCallback_click)
        else:
            print("No mode named:" + mode)
        cv2.startWindowThread()
        cv2.imshow(anno.window_name, img)
        while True:
            # By judging status, the program could be more responsive.
            if anno.deleting:
                if cv2.waitKey(10) == os:  # Backspace
                    del anno.annotation[anno.cur_file][anno.person_selected]
                    anno.deleting = False
                    anno.person_selected = -1
                    anno.joint_selected = ''
                    img = anno.img.copy()  # clear all skeleton drawn on the image
                    anno.plot_skeleton(img, anno.cur_file, thick=2)
                    cv2.imshow(anno.window_name, img)
            else:
                key = cv2.waitKey(10)
                if key == 13 or key == 3:  # Enter
                    anno.joint_selected = False
                    anno.revise()
                    idx += 1
                    break
                elif key == 2:
                    if idx >= 1:
                        idx -= 1
                    break
                elif key == 32:  # Space
                    global add_joint_num
                    anno.adding = True
                    anno.add_joint_num = int(input("Please input the joint number:")) + 1

                    # pop_box()
                    # anno.add_joint_num = int(anno.add_joint_num) + 1
                    # add_joint_num = ''

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
    for idx, anno.cur_file in enumerate(anno.data_files):
        black = np.zeros((360, 640, 3))
        cv2.namedWindow('Black')
        anno.plot_skeleton(black, anno.cur_file, 2)
        cv2.startWindowThread()
        cv2.imshow('Black', black)
        while True:
            if cv2.waitKey(10) & 0xFF == ord('\r'):
                break


def distribute(datadir):
    for root, dirs, _ in os.walk(datadir):
        if root == datadir:
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'joint_point.txt')):
                    os.remove(os.path.join(root, dir, 'joint_point.txt'))
                if os.path.exists(os.path.join(root, dir, 'refined.txt')):
                    os.remove(os.path.join(root, dir, 'refined.txt'))
    with open(os.path.join(datadir, 'annotation_all.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' : ')
            name = line[0]
            point = line[1]
            name = name.split('\\')
            folder = name[0]
            jpg = name[1]
            with open(os.path.join(datadir, folder, 'joint_point.txt'), 'a')as subf:
                subf.write(jpg + ' : ' + point)
    print('Distribution completed!')

def assemble(datadir):
    if os.path.exists(os.path.join(datadir, 'refined_all.txt')):
        os.remove(os.path.join(datadir, 'refined_all.txt'))
    for root, dirs, _ in os.walk(datadir):
        if root == datadir:
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'refined.txt')):
                    with open(os.path.join(datadir, 'refined_all.txt'), 'a') as fw:
                        with open(os.path.join(root, dir, 'refined.txt'), 'r') as fr:
                            lines = fr.readlines()
                            for line in lines:
                                line = line.split(' : ')
                                name = line[0]
                                point = line[1]
                                folder = dir + '\\' + name
                                fw.write(folder + ' : ' + point)
    print('Assembling completed!')

if __name__ == "__main__":
    # anno_dir = '/Users/midora/Desktop/MW-Pose-old/section_del'
    # dir = 'F:/capref/'
    dir = '/Users/midora/Documents/MW-Pose-dataset/dataset/_12.0'
    # dir = 'D:\\Documents\\Source\\MW-Pose-dataset\\dataset\\_12.0'
    # move_anno(anno_dir, dir)
    # pop_box()0
    # radar_out(dir)
    refine(dir, 'drag', thread=0, os=win)
    # distribute(dir)
    # assemble(dir)
    # print('Completed!')
    exit()
