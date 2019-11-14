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
#                               valid_deseqnet.py
#
#   Validation procedure of deseqnet.
#
#   Shrowshoo-Young 2019-2, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

import torch
import argparse
import os
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad
from src.utils.pose_decoder import *
from src.utils.evaluation import *
from src.model.deseqnet import DeSeqNetProj, DeSeqNetTest
from src.dataset import deSeqNetLoader
from src.utils import logger, imwrite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument('--data_path', type=str, default="../data/capref2", help="directory of dataset")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--pretrained_weights", type=str, default="checkpoints/deseqnettest_220.pth",
      help="if specified starts from checkpoint model")
    # parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    with open(os.path.join(opt.data_path, 'keypoint.names'), 'r') as namefile:
        keypointnames = namefile.readlines()
        keypointnames = [line.rstrip() for line in keypointnames]

    logger_val = logger('logger_val_10.14.txt')
    logger_valtag = keypointnames.copy()
    logger_valtag.append('average')
    logger_val.set_tags(logger_valtag)

    # Initiate model
    model = DeSeqNetTest().to(device)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))

    validset = deSeqNetLoader(opt.data_path, valid=True)
    validloader = torch.utils.data.DataLoader(
        validset,
        # batch_size=opt.batch_size,
        batch_size = 1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Validation progress
    model.eval()
    print("\n---- Evaluating Model ----\n")
    # Evaluate the model on the validation set

    # PCKh(8x18) stores the percentage of correct keypoint (head) during validation.
    # They are respectively     PCKh@0.05               PCKh@0.10
    #                           PCKh@0.20               PCKh@0.30
    #                           PCKh@0.40               PCKh@0.50
    #                           PCKh@0.75               PCKh@1.00
    n = len(keypointnames)
    PCKh = np.zeros([8, n + 1])
    thre = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5]

    for batch_i, (_, val_signal, GT) in enumerate(validloader):

        val_signal = Variable(val_signal.to(device), requires_grad=False)

        # start_time = time.time()
        val_outputs = model(val_signal)
        # end_time = time.time()
        # print(end_time - start_time)

        val_outputs = val_outputs.cpu()
        pred = pose_decode(val_outputs)
        print('[Valid, Batch %d/%d]\n' % (batch_i, len(validloader)))
        for i in range(8):
            PCKh[i, 0:-1] = PCKh[i, 0:-1] + eval_pckh(pred, GT, n, thre[i])

    PCKh[:] = PCKh[:]/len(validloader)
    for i in range(8):
        PCKh[i, -1] = np.sum(PCKh[i, 0:-1])/n
        logger_val.append(list(PCKh[i, :]))

        log_str = "PCKh@%f" % thre[i]
        for keypoint_i, name in enumerate(keypointnames):
            log_str += '[%s, %f]' % (name, PCKh[i, keypoint_i])

        log_str += '[average, %f]\n' % PCKh[i, -1]
        print(log_str)

    logger_val.close()