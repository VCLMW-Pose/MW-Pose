'''
    Created on Mon Mar 25 11:30 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   :

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
from PIL import Image
import matplotlib.pyplot as plt

__all__ = ['Annotation']

class Annotation:
    def __init__(self, dir):
        """
        :param dir: Directory of folder for pre-annotated data
                    e.g. /Users/midora/Desktop/MW-Pose/section_del/_1.0
        """
        self.dir = dir
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

    def __len__(self):
        return len(self.annotation)

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


def tag(name):
    pass


def load_img(dir, file):
    img = Image.open(os.path.join(dir, file))
    plt.figure("Image")
    plt.imshow(img)
    plt.axis('on')
    plt.title('image')
    plt.show()

def plot_skeleton(img, coor, thick):
    '''
        Args:
            window_name: (string)
            img: (ndarray) image for annotating
            coor: (list or tuple) shape (16, 2)
            thick: (int) thick of the line
            key: (int) the length of time the window stays
    '''
    if not ((coor[0][0] == 0 and coor[0][1] == 0) or (coor[1][0] == 0 and coor[1][1] == 0)):
        img = plt.plot(img, coor[0], coor[1], (181, 102, 60), thick)
    if not ((coor[1][0] == 0 and coor[1][1] == 0) or (coor[2][0] == 0 and coor[2][1] == 0)):
        img = plt.plot(img, coor[1], coor[2], (250, 203, 91), thick)
    if not ((coor[2][0] == 0 and coor[2][1] == 0) or (coor[6][0] == 0 and coor[6][1] == 0)):
        img = plt.plot(img, coor[2], coor[6], (35, 98, 177), thick)
    if not ((coor[3][0] == 0 and coor[3][1] == 0) or (coor[6][0] == 0 and coor[6][1] == 0)):
        img = plt.plot(img, coor[3], coor[6], (35, 98, 177), thick)
    if not ((coor[3][0] == 0 and coor[3][1] == 0) or (coor[4][0] == 0 and coor[4][1] == 0)):
        img = plt.plot(img, coor[3], coor[4], (66, 218, 128), thick)
    if not ((coor[4][0] == 0 and coor[4][1] == 0) or (coor[5][0] == 0 and coor[5][1] == 0)):
        img = plt.plot(img, coor[4], coor[5], (62, 121, 58), thick)
    if not ((coor[6][0] == 0 and coor[6][1] == 0) or (coor[7][0] == 0 and coor[7][1] == 0)):
        img = plt.plot(img, coor[6], coor[7], (23, 25, 118), thick)
    if not ((coor[7][0] == 0 and coor[7][1] == 0) or (coor[8][0] == 0 and coor[8][1] == 0)):
        img = plt.plot(img, coor[7], coor[8], (152, 59, 98), thick)
    if not ((coor[8][0] == 0 and coor[8][1] == 0) or (coor[9][0] == 0 and coor[9][1] == 0)):
        img = plt.plot(img, coor[8], coor[9], (244, 60, 166), thick)
    if not ((coor[8][0] == 0 and coor[8][1] == 0) or (coor[12][0] == 0 and coor[12][1] == 0)):
        img = plt.plot(img, coor[8], coor[12], (244, 59, 166), thick)
    if not ((coor[11][0] == 0 and coor[11][1] == 0) or (coor[12][0] == 0 and coor[12][1] == 0)):
        img = plt.plot(img, coor[11], coor[12], (51, 135, 239), thick)
    if not ((coor[10][0] == 0 and coor[10][1] == 0) or (coor[11][0] == 0 and coor[11][1] == 0)):
        img = plt.plot(img, coor[10], coor[11], (35, 98, 177), thick)
    if not ((coor[8][0] == 0 and coor[8][1] == 0) or (coor[13][0] == 0 and coor[13][1] == 0)):
        img = plt.plot(img, coor[8], coor[13], (244, 59, 166), thick)
    if not ((coor[13][0] == 0 and coor[13][1] == 0) or (coor[14][0] == 0 and coor[14][1] == 0)):
        img = plt.plot(img, coor[13], coor[14], (49, 56, 218), thick)
    if not ((coor[14][0] == 0 and coor[14][1] == 0) or (coor[15][0] == 0 and coor[15][1] == 0)):
        img = plt.plot(img, coor[14], coor[15], (23, 25, 118), thick)
    return img

def annotate(dir):
    if os.path.exists(os.path.join(dir, 'joint_point.txt')):
        with open(os.path.join(dir, 'joint_point.txt'), 'r') as f:
            lines = f.reandlines()
    for _, dirs, _ in os.walk(dir, topdown=True):
        pass

if __name__ == "__main__":
    dir = 'res/'
    anno = Annotation(dir)
    for file in anno.data_files:
        load_img(dir, file)
        break
    print('Completed!')
