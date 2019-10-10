'''
    Created on Thu Oct 10 10:44 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Thu Oct 10 11:34 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import numpy as np
import cv2

parts = ['nose', 'neck', 'rShoulder',
         'rElbow', 'rWrist', 'lShoulder',
         'lElbow', 'lWrist', 'rHip', 'rKnee',
         'rAnkle', 'lHip', 'lKnee', 'lAnkle',
         'rEye', 'lEye', 'rEar', 'lEar']


def plot_skeleton(img, output, thick=2):
    '''
        Args:
            window_name:    (string)
            img:            (PILImage) image for annotating
            data_file:      (string) file name e.g. 1551601527845.jpg
            thick:          (int) thick of the line
            key:            (int) the length of time the window stays
    '''
    jointscoor = {}
    for i in range(len(parts)):
        jointscoor[parts[i]] = (output[i][0], output[i][1])
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
    for joint in parts:
        if jointscoor[joint] != (-1, -1):
            img = cv2.circle(img, jointscoor[joint], 3, (68, 147, 200), -1)
    return img

def decoder(output):
    """
    Arg:
        output: (18x64x64 ndarray) heatmap
    Return:
        op_np: (18x2 ndarray) joints coordinate in hea*tmap
        op_list: (18x2 list) joints coordinate in image
    """
    op_np = np.zeros((len(parts), 2), dtype=int)
    for part in range(len(parts)):
        part_output = output[:, part, :, :]
        if part_output.max() >= 0.2:
            print(np.where(part_output == part_output.max()))
            op_np[part][0] = np.where(part_output == part_output.max())[1]
            op_np[part][1] = np.where(part_output == part_output.max())[2]
        else:
            op_np[part][0] = -1
            op_np[part][1] = -1
    op_list = [[0, 0]] * len(parts)  #For drawing
    for part in range(len(parts)):
        op_list[part] = [op_np[part][1] * 10, op_np[part][0] * 10]  # *640/64 scaling to the image size
    return op_np, op_list



if __name__ == "__main__":
    output = np.random.random((18, 64, 64))
    black = np.zeros((360, 640, 3))
    cv2.namedWindow('Black')
    output_np, output_list = decoder(output)
    plot_skeleton(black, output_list, thick=2)
    cv2.imshow('Black', black)
    while True:
        if cv2.waitKey(10) & 0xFF == ord('\r'):
            break