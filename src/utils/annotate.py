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
        img = plt.plot(img, joints['rank'], joints['rkne'], color='#B5663C', linewidth=thick)
        img = plt.plot(img, joints['rkne'], joints['rhip'], color='#FACB5B', linewidth=thick)
        img = plt.plot(img, joints['rhip'], joints['pelv'], color='#2362B1', linewidth=thick)
        img = plt.plot(img, joints['lhip'], joints['pelv'], color='#2362B1', linewidth=thick)
        img = plt.plot(img, joints['lhip'], joints['lkne'], color='#42DA80', linewidth=thick)
        img = plt.plot(img, joints['lkne'], joints['lank'], color='#3E793A', linewidth=thick)
        img = plt.plot(img, joints['pelv'], joints['thrx'], color='#171976', linewidth=thick)
        img = plt.plot(img, joints['thrx'], joints['neck'], color='#983B62', linewidth=thick)
        img = plt.plot(img, joints['neck'], joints['head'], color='#F43CA6', linewidth=thick)
        img = plt.plot(img, joints['neck'], joints['rsho'], color='#F43BAC', linewidth=thick)
        img = plt.plot(img, joints['relb'], joints['rsho'], color='#3387EF', linewidth=thick)
        img = plt.plot(img, joints['rwri'], joints['relb'], color='#2362B1', linewidth=thick)
        img = plt.plot(img, joints['neck'], joints['lsho'], color='#F43BAC', linewidth=thick)
        img = plt.plot(img, joints['lsho'], joints['lelb'], color='#3138DA', linewidth=thick)
        img = plt.plot(img, joints['lelb'], joints['lwri'], color='#171976', linewidth=thick)
        return img

def tag(name):
    pass


def load_img(dir, file):
    img = Image.open(os.path.join(dir, file))
    plt.figure("Image")
    plt.imshow(img)
    plt.axis('on')
    plt.title('image')
    
    plt.show()





def annotate(dir):
    if os.path.exists(os.path.join(dir, 'joint_point.txt')):
        with open(os.path.join(dir, 'joint_point.txt'), 'r') as f:
            lines = f.reandlines()
    for _, dirs, _ in os.walk(dir, topdown=True):
        pass


if __name__ == "__main__":
    dir = '/Users/midora/Desktop/MW-Pose/section_del/_7.0'
    anno = Annotation(dir)
    for file in anno.data_files:
        load_img(dir, file)
        img = Image.open(os.path.join(dir, file))
        plt.figure("Image")
        plt.imshow(img)
        plt.axis('on')
        plt.title('image')
        anno.plot_skeleton(img, file, 0.2)
        plt.show()
        break
    print('Completed!')
