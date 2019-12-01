'''
    Created on Thu Oct 10 10:44 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Thu Oct 10 11:34 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)

    while 1:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


