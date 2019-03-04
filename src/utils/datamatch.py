'''
    Created on Sun Mar 3 19:20 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Mon Mar 4 16:16 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
import shutil
from random import random, seed
from matplotlib import pyplot as plt
import numpy as np


def matching(data_dir, max_err):
    """
    This function is designed to match walabot data with optical data
            having the least time delay(less than max_err).
    :param data_dir: directory of the top folder of data
    :param max_err: maximum acceptable time delay
    """
    matched_num = 0
    for _, dirs, _ in os.walk(data_dir, topdown=True):
        with open(data_dir + "/time_info.txt", 'a') as f:
            for dir in dirs:
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
                        elif file[-1] >= '0' and file[-1] <= '9':
                            walabots.append(file)
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
                            shutil.copy(root + '/' + walabot, dst)
                            shutil.copy(root + '/' + matched, dst)
                            f.writelines([walabot, '\t', matched[:-4], '\t', str(cur_err)[:7], '\n'])
                            matched_num += 1
        break  # Only traverse top directory
    print("%d data has been matched." % (matched_num))


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
    x = np.array(x)
    delays = np.array(delays)
    plt.title('Time Delay Scatter', fontsize=16, loc='left', color='g')
    plt.xlim((-0.2, 1.2))
    plt.ylim((0, max_err))
    plt.ylabel('Difference (ms)', fontsize=10, color='b')
    plt.scatter(x, delays, c='b')
    plt.savefig(data_dir + "/Scatter.png")
    plt.show()


if __name__ == "__main__":
    matching("/Users/midora/Desktop/MW-Pose/datacontainer", 0.01)
    analysis("/Users/midora/Desktop/MW-Pose/datacontainer", 0.01)
    print("Completed!")