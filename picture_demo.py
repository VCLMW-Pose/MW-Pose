import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from network.rtpose_vgg import get_model
from network.post import decode_pose
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network import im_transform
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from openpose_utils import *
dir = 'D:\\Documents\\Source\\MW-Pose\\'
datadir = 'D:\\Documents\\Source\\MW-Pose\\test\\'
weight_name = 'D:\\Documents\\Source\\MW-Pose\\openpose\\pose_model.pth'

loader = Loader(datadir)
saver = Saver(datadir)

model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()


for i, fname in enumerate(loader):
    oriImg = cv2.imread(fname) # B,G,R order
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of original image
    multiplier = get_multiplier(oriImg)

    with torch.no_grad():
        orig_paf, orig_heat = get_outputs(
            multiplier, oriImg, model,  'rtpose')
          
        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                model, 'rtpose')

        # compute averaged heatmap and paf
        paf, heatmap = handle_paf_and_heat(
            orig_heat, flipped_heat, orig_paf, flipped_paf)
            
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    canvas, to_plot, joint_list, person_to_joint_assoc = decode_pose(
        oriImg, param, heatmap, paf)
    saver.crawl(fname, joint_list, person_to_joint_assoc)
    cv2.imwrite(dir + 'done\\' + str(i)+'.png', to_plot)
    print('%d images have been annotated!' % i)
print('Annotation completed!')
saver.distribute()
exit()

