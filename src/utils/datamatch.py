'''
    Created on Sun Mar 3 19:20 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Tue Mar 26 22:55 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
import shutil
from random import random, seed
from matplotlib import pyplot as plt
import numpy as np


def matching(data_dir, max_err, rm_ori_file=False):
    """
    This function is designed to match walabot data with optical data
            having the least time delay(less than max_err).
    :param data_dir: directory of the top folder of data
    :param max_err: maximum acceptable time delay
    :param rm_ori_file: Whether to remove the original files
    """
    tt_jpg_num = 0
    tt_walbolot_num = 0
    tt_matched_num = 0  # Number of data files matched in all the folders
    for _, dirs, _ in os.walk(data_dir, topdown=True):
        with open(data_dir + "/time_info.txt", 'a') as f:
            for dir in dirs:
                matched_num = 0  # Number of data files matched in each folder
                if dir[0] == '_' or os.path.exists(data_dir + "/_" + dir):
                    continue
                _dir = data_dir + "/_" + dir  # Target folder
                os.mkdir(_dir)
                for root, _, files in os.walk(data_dir + '/' + dir):
                    jpgs = []
                    walabots = []
                    for file in files:
                        # Do not search jpg ,txt and hidden files
                        if file[-1] == 'g':
                            jpgs.append(file)
                            tt_jpg_num += 1
                        elif file[-1] >= '0' and file[-1] <= '9':
                            walabots.append(file)
                            tt_walbolot_num += 1
                    jpgs.sort()
                    walabots.sort()
                    pos = 0
                    for walabot in walabots:
                        matched = ""
                        # Name of walabot data: Universal Time displayed in millisecond
                        # First 10 bit: second
                        # Last 3 bit millisecond
                        sec = float(walabot[:10])
                        milsec = float(walabot[10:])
                        time = sec + milsec / 1000
                        cur_err = max_err
                        while pos < len(jpgs):
                            jpg = jpgs[pos]
                            # Name of optical data: Universal Time displayed in second
                            _time = float(jpg[:-4])
                            if abs(time - _time) <= cur_err:
                                cur_err = abs(time - _time)
                                matched = jpg
                            if (_time - time > cur_err):
                                break
                            pos += 1
                        if matched != "":
                            dst = _dir + '/' + walabot
                            os.mkdir(dst)
                            if rm_ori_file:
                                shutil.move(root + '/' + walabot, dst)
                                shutil.move(root + '/' + matched, dst)
                                os.rename(os.path.join(dst, matched), os.path.join(dst, walabot + '.jpg'))
                            else:
                                shutil.copy(root + '/' + walabot, dst)
                                shutil.copy(root + '/' + matched, dst)
                                os.rename(os.path.join(dst, matched), os.path.join(dst, walabot + '.jpg'))
                            f.writelines([walabot, '\t', matched[:-4], '\t', str(cur_err), '\n'])
                            matched_num += 1
                            tt_matched_num += 1
                print("%d data has been matched in %s." % (matched_num, dir))
        break  # Only traverse top directory
    print("%d data has been matched totally." % tt_matched_num)
    print("%d optical data files are gathered." % tt_jpg_num)
    print("%d walabot signal files are gathered." % tt_walbolot_num)


def analysis(data_dir, max_err):
    """
    Using the time_info text file to sketch the time delay distribution.
    Running "matching" function before using this function is recommended.
    :param data_dir: directory of the top folder of data
    :param max_err: maximum acceptable time delay
    """
    dir = data_dir + "/time_info.txt"
    if not os.path.exists(data_dir + "/time_info.txt"):
        print("There is no time data for analysis!")
        return
    with open(dir, 'r') as f:
        lines = f.readlines()
    delays = []
    x = []
    for line in lines:
        line = line.rstrip()
        delay = line.split('\t')[2]
        x.append(random())
        delays.append(float(delay))
    matched = len(x)
    x = np.array(x)
    delays = np.array(delays)
    plt.title('Time Delay Scatter', fontsize=16, loc='left', color='g')
    plt.xlim((0, 1))
    plt.ylim((0, max_err))
    plt.ylabel('Difference (s)', fontsize=10, color='b')
    plt.scatter(x, delays, c='b', s=2)
    plt.text(0.6, 0.0102, 'Totally matched data:%d' % matched, color='b')
    plt.savefig(data_dir + "/Scatter.png")
    plt.show()


def frame_analysis(data_dir):
    """
    :param data_dir: directory of the top folder of data
    """

    for _, dirs, _ in os.walk(data_dir, topdown=True):
        for dir in dirs:
            if dir[0] != '_':
                continue
            walabots = []
            for root, subdirs, _ in os.walk(os.path.join(data_dir, dir), topdown=True):
                for subdir in subdirs:
                    walabots.append(float(subdir))
                walabots.sort()
                break
            x = np.arange(0, len(walabots), 1)
            walabots = np.array(walabots)
            walabots = walabots - walabots[0]
            walabots = walabots / 1000
            plt.title('Frame Anaslysis ' + dir, fontsize=16, loc='left', color='g')
            plt.xlim((0, len(walabots)))
            plt.ylim((0, walabots[-1]))
            plt.ylabel('Interval (s)', fontsize=10, color='b')
            plt.scatter(x, walabots, c='b', s=10)
            plt.savefig(data_dir + '/' + dir + ".png")
            plt.show()
        break  # Only traverse top directory


def frame_analysis_anno(data_dir):
    """
    :param data_dir: directory of the top folder of data
    """

    for _, dirs, _ in os.walk(data_dir, topdown=True):
        dirs.sort()
        for dir in dirs:
            if dir[0] != '_':
                continue
            walabots = []
            for root, _, files in os.walk(os.path.join(data_dir, dir), topdown=True):
                for file in files:
                    if (file[-4:] == '.jpg'):
                        walabots.append(float(file[:-4]))
                walabots.sort()
                break
            x = np.arange(0, len(walabots), 1)
            walabots = np.array(walabots)
            walabots = walabots - walabots[0]
            walabots = walabots / 1000
            plt.title('Frame Anaslysis ' + dir, fontsize=16, loc='left', color='g')
            plt.xlim((0, len(walabots)))
            plt.ylim((0, walabots[-1]))
            plt.ylabel('Interval (s)', fontsize=10, color='b')
            plt.scatter(x, walabots, c='b', s=10)
            plt.savefig(data_dir + '/' + dir + ".png")
            plt.show()
        break  # Only traverse top directory

def move_img(dir_orig, dir_dest):
    """
    Description: To move image to dataset package
    :param dir_orig:
    :param dir_dest:
    :return: 
    """
    jpgs = []
    fulljpgs = []
    tt_jpg_num = 0
    for root, _, files in os.walk(dir_orig):
        for file in files:
            # Do not search jpg ,txt and hidden files
            if file[-4:] == '.jpg':
                jpgs.append(file)
                fulljpgs.append(os.path.join(root, file))
                tt_jpg_num += 1
    with open(os.path.join(dir_dest, 'joint_point.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' : ')
            name = line[0].split('\\')[1]
            for i in range(len(jpgs)):
                if name == jpgs[i]:
                    shutil.copy(fulljpgs[i], os.path.join(dir_dest, 'images'))


if __name__ == "__main__":
    # matching("/Users/midora/Desktop/MW-Pose/datacontainer", 0.01, False)
    # analysis("/Users/midora/Desktop/MW-Pose/datacontainer", 0.01)
    # frame_analysis("/Users/midora/Desktop/MW-Pose/datacontainer")
    # frame_analysis_anno('/Users/midora/Desktop/MW-Pose/section_del')
    dir_orig = 'D:\\Documents\\Source\\MW-Pose-dataset-old\\dataset-new'
    dir_dest = 'D:\\Documents\\Source\\MW-Pose\\data\\matched-new'
    # move_img(dir_orig, dir_dest)
    matching(dir_orig, 0.01, False)

    print("Completed!")
    exit()
