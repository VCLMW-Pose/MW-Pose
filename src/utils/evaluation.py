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

##################################################################################
#                                  evaluation.py
#
#   Shrowshoo-Young 2019-10, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

import numpy as np


def eval_pckh(pred, gt, keypointnum, range):
    head_size = get_head_size(gt)
    dist = get_dist_pckh(pred, gt, head_size, keypointnum)

    n = pred.shape[0]
    pck = np.zeros(18)

    for i in range(18):
        pck[i] = sum(dist[:, i] <= range)/n
    return pck


def get_head_size(target):
    n = target.shape[0]
    head_size = np.sqrt(4*(target[:, 0, 0] - target[:, 1, 0])**2 + 4*(target[:, 0, 1] - target[:, 1, 1])**2)

    for i in range(n):
        if sum(target[i, 0:2, :] == -1):
            head_size[i] = 20

    return head_size


def get_dist_pckh(pred, gt, refdist, keypointnum):
    # pred nx18x2
    # dist is a nx18 matrix
    n = pred.shape[0]
    dist = np.zeros(n, keypointnum)

    for i in range(n):
        dist[i, :] = np.sqrt((pred[i, :, 0] - gt[i, :, 0])**2 + (pred[i, :, 1] - gt[i, :, 1])**2)/refdist[i, :]

    return dist