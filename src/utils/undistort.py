#-*- coding = utf-8 -*-
"""
# Copyright (c) 2018-2019, Shrowshoo-Young All Right Reserved.
#
# This programme is developed as free software for the investigation of human
# pose estimation using RF signals. Redistribution or modification of it is
# allowed under the terms of the GNU General Public Licence published by the
# Free Software Foundation, either version 3 or later version.
#
# Redistribution and use in source or executable programme, with or without
# modification is permitted provided that the following conditions are met:
#
# 1. Redistribution in the form of source code with the copyright notice
#    above, the conditions and following disclaimer retained.
#
# 2. Redistribution in the form of executable programme must reproduce the
#    copyright notice, conditions and following disclaimer in the
#    documentation and\or other literal materials provided in the distribution.
#
# This is an unoptimized software designed to meet the requirements of the
# processing pipeline. No further technical support is guaranteed.
"""

import cv2
import numpy as np
import glob

# Intrinsic camera parameter
camera_matrix = np.array([[511.507088947061,	0,	                0],
                          [0.252208503036723,	510.941228363921,	0],
                          [305.319150841243,	156.680409485688,	1]])
camera_matrix = np.transpose(camera_matrix)

# Distortion coefficienets
# dist_coeff = np.array([0, 0, 0, 0, 0])
# dist_coeff = np.array([-0.448119640947490, 0.234356931336826, -0.001516246547021,
#                        0.001820274760021])
dist_coeff = np.array([-0.440877497104674,	0.220637195683300, -0.00151624654702094,
                       0.00182027476002143])
# dist_coeff = np.array([-0.001516246547021, 0.001820274760021, -0.448119640947490,
#                        0.234356931336826])


def undistort(img):
    # Find dimension of image
    w, h = img.shape[:2]

    # Compute new camera parameter matrix, and solve undistorted image
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (h, w), 0, (h, w))
    img = cv2.undistort(img, camera_matrix, dist_coeff, None, new_camera_matrix)
    return img


def undistort_test():
    img = cv2.imread('F:/captureNov15/1/0054.jpg')
    img = undistort(img)

    cv2.imshow('test', img)
    cv2.waitKey(10000)


if __name__ == "__main__":
    undistort_test()
