# -*- coding: utf-8 -*-
'''
    Created on Thu Sep 20 21:25 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Tue Oct 2 15:02 2018

South East University Automation College, 211189 Nanjing China
'''

import cv2
from src.train_mpii import *
from copy import deepcopy
from src.utils import draw, insert_img
import torch
import time
from src.dataset.mpii import Mpii
from src.dataset.dataloader import MpiiDataset


class Estimator:
    def __init__(self, model, camera=False):
        self.model = model
        self.camera = camera
        self.parts = ['rank', 'rkne', 'rhip',
                      'lhip', 'lkne', 'lank',
                      'pelv', 'thrx', 'neck', 'head',
                      'rwri', 'relb', 'rsho',
                      'lsho', 'lelb', 'lwri']

    def test(self, dataset):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for i, (idx, data, img, target) in enumerate(data_loader):
            data = Variable(data.type(Tensor))
            target = Variable(target.type(Tensor), requires_grad=False)
            # print(target.shape)
            gt_np = dataset.get_parts(int(idx))
            img = np.array(img[0])
            # Using ground truth
            img_gt = deepcopy(img)
            # Using heatmap of ground truth
            img_tg = deepcopy(img)
            output = self.model(data)
            op_np = np.zeros((16, 2), dtype=int)
            tg_np = np.zeros((16, 2), dtype=int)
            for part in range(len(self.parts)):
                part_output = output[0, part + len(self.parts), :, :]
                part_target = target[0, part + len(self.parts), :, :]
                if part_output.max() >= 0.4:
                    op_np[part][0] = np.where(part_output == part_output.max())[0][0]
                    op_np[part][1] = np.where(part_output == part_output.max())[1][0]
                if part_target.max() != 0:
                    tg_np[part][0] = np.where(part_target == part_target.max())[0][0]
                    tg_np[part][1] = np.where(part_target == part_target.max())[1][0]
            op = [[0, 0]] * len(self.parts)
            gt = [[0, 0]] * len(self.parts)
            tg = [[0, 0]] * len(self.parts)
            for part in range(len(self.parts)):
                op[part] = op_np[part][0] * 4, op_np[part][1] * 4
                gt[part] = int(gt_np[part][0]), int(gt_np[part][1])
                tg[part] = int(tg_np[part][0] * 4), int(tg_np[part][1] * 4)

            draw(img, op, 3)
            draw(img_gt, gt, 3)
            draw(img_tg, tg, 3)

            cv2.imshow('Estimator', img)
            cv2.imshow('Ground Truth', img_gt)
            cv2.imshow('Target', img_tg)

            cv2.waitKey(0)

    def estimate(self, img, bbox, thresh=0.4):
        '''
            Args:
                img     : (ndarray) original image got from cv2.imread()
                bbox    : (list) [x1, y1, x2, y2]
            Return:
                tg      :(list 16)(tuple 2) joint point of original image
                        [(x1, y1), (x2, y2), ... , (x16, y16)]
        '''
        dx = 0
        dy = 0
        height, width = img.shape[:2]
        [x1, y1, x2, y2] = bbox
        new_img = img[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]
        if x1 > 0:
            dx += x1
        if y1 > 0:
            dy += y1
        new_height, new_width = new_img.shape[:2]
        max_ = max(new_width, new_height)
        scale = max_ / 256
        if new_height > new_width:
            left = (new_height - new_width) // 2
            right = abs(new_height - new_height - left)
            new_img = cv2.copyMakeBorder(new_img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            # lts = torch.ones(3, max, left) * 128 / 255  # left tensor
            # rts = torch.ones(3, max, right) * 128 / 255
            # new_img = torch.cat((lts, new_img, rts), 2)
            dx -= left
        elif new_height < new_width:
            top = (new_width - new_height) // 2
            bottom = abs(new_width - new_height - top)
            new_img = cv2.copyMakeBorder(new_img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            # tts = torch.ones(3, top, max) * 128 / 255  # top tensor
            # bts = torch.ones(3, bottom, max) * 128 / 255
            # new_img = torch.cat((tts, new_img, bts), 1)
            dy -= top
        if max_ > 256:
            new_img = cv2.resize(new_img, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            new_img = cv2.resize(new_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        new_img = new_img.swapaxes(1, 2).swapaxes(0, 1)
        new_img = torch.from_numpy(new_img).float() / 255
        data = new_img.unsqueeze(0)

        if torch.cuda.is_available():
            data = data.cuda()

        output = self.model(data)
        op_np = np.zeros((16, 2), dtype=int)
        for part in range(len(self.parts)):
            part_output = output[0, part + len(self.parts), :, :]
            if part_output.max() != 0:  # and part_output.max() >= thresh:
                op_np[part][0] = np.where(part_output == part_output.max())[0][0]
                op_np[part][1] = np.where(part_output == part_output.max())[1][0]
        # print('target = ', op_np)
        op = [[0, 0]] * len(self.parts)
        for part in range(len(self.parts)):
            op[part] = int(op_np[part][0] * 4 * scale + dx), int(op_np[part][1] * 4 * scale + dy)
        return op

    def tg_check(self, dataset):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for i, (idx, _, img, target) in enumerate(data_loader):
            target = Variable(target.type(Tensor), requires_grad=False)
            img = np.array(img[0])
            # Using heatmap of ground truth
            tg_np = np.zeros((16, 2), dtype=int)
            for part in range(len(self.parts)):
                part_target = target[0, part + len(self.parts), :, :]
                if part_target.max() != 0:
                    tg_np[part][0] = np.where(part_target == part_target.max())[0][0]
                    tg_np[part][1] = np.where(part_target == part_target.max())[1][0]
            # print('target = ', tg_np)
            tg = [[0, 0]] * len(self.parts)
            for part in range(len(self.parts)):
                tg[part] = int(tg_np[part][0] * 4), int(tg_np[part][1] * 4)
            # draw(img, tg, 3)
            insert = cv2.imread('/Users/midora/Downloads/01300542846491152239525449451.jpg')
            start = time.time()
            insert_img(img, insert, 9, tg)
            t = time.time() - start
            print(t)
            cv2.imshow('Target', img)
            cv2.waitKey(0)

    def gt_check(self):
        mpii = MpiiDataset(FolderPath, Annotation)
        for i in range(len(mpii)):
            if mpii.containers[i].istrain:
                dir, num_pp, gt_np = mpii[i]
                gt = [[[0, 0]] * 16] * num_pp
                img = cv2.imread(dir)
                # doge = cv2.imread('/Users/midora/Desktop/Python/HPE/res/doge.jpg')
                # doge = cv2.imread('/Users/midora/Desktop/Python/HPE/res/pikachu.jpg')
                for pp in range(num_pp):
                    for part in range(16):
                        gt[pp][part] = int(gt_np[pp][part][0]), int(gt_np[pp][part][1])
                    draw(img, gt[pp], 2)
                    # insert_img(img, doge, 'lsho', gt[pp])
                cv2.imshow('Ground Truth', img)
                cv2.waitKey(0)


if __name__ == "__main__":
    weight_file_name = WeightPath + "stacked_hourglass.pkl"
    # Dataset
    # dataset = MpiiDataSet_sig(FolderPath, Annotation, if_train=False)
    # Model
    model = StackedHourglass(16)
    if os.path.isfile(weight_file_name):
        model.load_state_dict(torch.load(weight_file_name))
    if torch.cuda.is_available():
        model.cuda()

    # Estimator
    estimator = Estimator(model)
    # estimator.test(dataset)
    # estimator.tg_check(dataset)
    estimator.gt_check()
