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
#                               train_deseqnet.py
#
#   Training procedure of deseqnet.
#
#   Shrowshoo-Young 2019-2, shaoshuyangseu@gmail.com
#   South East University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import grad
from src.model import denseSequentialNet
from src.dataset import deSeqNetLoader
from src.utils import logger, imwrite

def train(args):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAGAN Training Experiment:")

    parser.add_argument('--dataset', default='D:/ShaoshuYang/MNIST/train', type=str, metavar='N', help='Directory '
                                                                                                        +'of data set')
    parser.add_argument('--testset', default='D:/ShaoshuYang/MNIST/test', type=str, metavar='N', help='Directory'
                                                                                                      +'of test set')
    parser.add_argument('--resume', default=False, type=bool, metavar='N', help='Resume traning or traning from '
                                                                                                        +'scratch')
    parser.add_argument('--model_name', default='DRAGAN', type=str, metavar='N', help='Name of model')
    parser.add_argument('--epoch', default=90, type=int, metavar='N', help='Epoches of training')
    parser.add_argument('--test_iter', default=10, type=int, metavar='N', help='Test set amount')
    parser.add_argument('--simple_num', default=100, type=int, metavar='N', help='Number of simples')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='Batch size of training')
    parser.add_argument('--save_dir', default='weights/', type=str, metavar='N', help='Directory of weight files')
    parser.add_argument('--result_dir', default='results/', type=str, metavar='N', help='Directory of result images')
    parser.add_argument('--log_dir', default='logger/', type=str, metavar='N', help='Directory to save logs')
    parser.add_argument('--test_log_dir', default='logger/', type=str, metavar='N', help='Directory to save test logs')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='N', help='Learning rate of generator')
    parser.add_argument('--beta1', default=0.5, type=float, metavar='N')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='N')
    parser.add_argument('--lambda_', default=0.25, type=float, metavar='N')
    parser.add_argument('--k', default=1, type=float, metavar='N')
    parser.add_argument('--gpu_mode', default=True, type=bool, metavar='N', help='Whether use GPU or not')
    parser.add_argument('--benchmark_mode', default=True, type=bool, metavar='N', help='Whether use cudnn')
    parser.add_argument('--z_dim', default=100, type=int, metavar='N', help='Dimension of input scalar')
    parser.add_argument('--x_dim', default=3, type=int, metavar='N', help='Dimension of generated result')
    parser.add_argument('--img_size', default=20, type=int, metavar='N', help='Scale of generated result')

    args = parser.parse_args()

    train(args)