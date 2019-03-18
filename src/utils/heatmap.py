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
#                                   heatmap.py
#
#   Definitions of data pre-processing toolkit.
#
#   Shrowshoo-Young 2019-2, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

import os
import random
import numpy as np

__all__ = ['putGaussianMap']

keyPointList = ["head", "neck", "lshoulder", "rshoulder", "lelbow", "relbow",
                "lhand", "rhand", "lhip", "rhip", "lknee", "rknee", "lankle", "rankle"]

# Specification of annotation files:
# [file name] [total number of people] [person id] [coordinates of joints]
#
# @brief: Every single row of annotation depicts the coordinates of key
# points of an individual within in the image named 'file name'. 'total number
# of people' denotes the number of people can be detected in corresponding
# image, each of them is given an id number to distinguish different individuals.
#
# @example:
#  [1551760709.9952404.jpg] [2] [1] [head: 100, 100] [neck: 100, 100] [ankle: 100, 100]...
#  [1551760709.9952404.jpg] [2] [2] [head: 200, 100] [neck: 200, 100] [ankle: 200, 100]...
#
# @order of key points: 1. head, 2. neck, 3. left shoulder, 4. right shoulder,
# 5. left elbow, 6. right elbow, 7. left hand, 8. right hand, 9. left hip,
# 10. right hip, 11. left knee, 12. right knee, 13. left ankle, 14. right ankle.
def readAnnotation(annoDirctory):
    file = open(annoDirctory, 'r')
    annotation = []
    content = file.readlines()
    iter = 0
    imageNum = 0

    while iter < len(content):
        annoString = content[iter].split()
        fileName = annoString[0].strip('[]')
        numOfPeople = int(annoString[1].strip('[]'))
        annotation = [annotation, {"fileName": fileName, "numOfPeople": numOfPeople, "head": [],
                                   "neck": [], "lshoulder": [], "rshoulder": [], "lelbow": [],
                                   "relbow": [], "lhand": [], "rhand": [], "lhip": [], "rhip": [],
                                   "lknee": [], "rknee": [], "lankle": [], "rankle": []}]

        for i in range(0, numOfPeople):
            annoString = content[iter].split()
            for m in range(3, 17):
                coord = annoString[m].strip('[]').split()
                annotation[imageNum][keyPointList[m - 3]].append([int(coord[1].rstrip(',')), int(coord[2])])
            iter += 1
        imageNum += 1
    file.close()
    return annotation

#
# @brief: Generate 2D gaussian heatmap as ground truth for key points detection.
# It produces as many heatmaps as the number of classes of key points.
def putGaussianMap(annotation):
    heatmap = []
    return heatmap