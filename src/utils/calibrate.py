'''
    Created on Mon Nov 18 21:49 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   :

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import numpy as np
import os
import cv2
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from math import sin, cos, pi


def Sphere2Cartesian(signal_dir):
    sig_file = open(signal_dir, mode='rb')
    data = np.fromfile(sig_file, dtype=np.int32)
    theta = torch.tensor([
        [cos(pi/4), sin(-pi/4), 0, 0],
        [sin(pi/4), cos(pi/4), 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    size_x = data[0]  # theta 46
    size_y = data[1]  # phi 45
    size_z = data[2]  # 59
    raw_img = np.array(data[3:]).reshape(size_x, size_y, size_z)
    # plt.imshow(raw_img.transpose(1, 2, 0))
    # plt.show()
    raw_img = torch.from_numpy(raw_img.astype(np.float32))
    N, C, D, H, W = raw_img.unsqueeze(0).unsqueeze(0).size()
    grid = F.affine_grid(theta.unsqueeze(0), torch.Size((N, C, D, W, H)))
    output = F.grid_sample(raw_img.unsqueeze(0).unsqueeze(0), grid)
    new_img_torch = output[0][0].numpy().transpose(0, 2, 1)
    npgrid = grid[0].numpy()
    for i in range(size_z):
        np.savetxt("/Users/midora/Documents/MW-Pose-dataset/capture_test/dets%d.txt" %i, new_img_torch[:, :, i], fmt='%d', delimiter='    ')
    # plt.imshow(new_img_torch.numpy().transpose(1, 2, 0))
    # plt.show()


if __name__ == '__main__':
    Sphere2Cartesian('/Users/midora/Documents/MW-Pose-dataset/captureNov15/2/0022')
    exit()
