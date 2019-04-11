# -*- coding: utf-8 -*-
'''
    Created on wed Sept 6 19:34 2018

    Author           : Shaoshu Yang, Heng Tan, Yue Han, Yu Du
    Email            : 13558615057@163.com
                       1608857488@qq.com
                       1015985094@qq.com
                       1239988498@qq.com
    Last edit date   : Tue Sept 25 21:07 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
How to implement a YOLO (v3) object detector from strach in
PyTorch: Part 3/4/5
'''

from __future__ import division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from cmath import sqrt



def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,
                                           keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(),
                                class_pred.float()), 1)

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]

            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]

            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))

                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_IOU(max_detections[-1], detections_class[1:])

                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else \
                torch.cat((output[image_i], max_detections))

    return output


def cpu(tensor):
    '''
        Args:
             tensor       : (tensor) input tensor
        Returns:
             Transfer tensors in GPU to CPU
    '''
    if tensor.is_cuda:
        return torch.FloatTensor(tensor.size()).copy_(tensor)

    else:
        return tensor


def pred_transform(prediction, in_dim, anchors, class_num, CUDA=True):
    '''
        Args:
             prediction   : (tensor) output of the network
             in_dim       : (int) input dimension
             anchors      : (tensor) describe anchors
             class_num    : (int) class numbers
             CUDA         : (bool) defines the accessibility to CUDA
                             and GPU computing
        Returns:

    '''
    batch_size = prediction.size(0)
    stride = in_dim // prediction.size(2)
    grid_size = prediction.size(2)
    bbox_attr = 5 + class_num
    anchor_num = len(anchors)

    # Transfroms to the prediction
    prediction = prediction.view(batch_size, bbox_attr * anchor_num,
                                 grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, anchor_num * grid_size * grid_size,
                                 bbox_attr)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Adding sigmoid to the x_coord, y__coord and objscore
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    # Add offset to the central coordinates
    offset_x = torch.FloatTensor(a).view(-1, 1)
    offset_y = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        offset_x = offset_x.cuda()
        offset_y = offset_y.cuda()

    offset_x_y = torch.cat((offset_x, offset_y), 1).repeat(1, anchor_num,
                                                           ).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += offset_x_y

    # Add log-space transforms
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Add sigmoid to classes possibility
    prediction[:, :, 5:5 + class_num] = torch.sigmoid(prediction[:, :, 5:5 +
                                                                         class_num])

    # Resize the detection map to the original image size
    prediction[:, :, :4] *= stride

    return prediction


# Adding objectness score thresholding and Non-maximal suppression
def write_results(prediction, confidence, class_num, nms_conf=0.4):
    '''
        Args:
             prediction  : (tensor) output tensor form darknet
             confidence  : (float) object score threshold
             class_num   : (num) number of classes
             num_conf    : (float) confidence of Non-maximum suppresion
        Returns:
             Results after confidence threshold and Non-maximum suppresion
             process
    '''
    # Set the attributes of a bounding-box to zero when its score
    # is below the threshold
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Transform the bx, by, bw, bh to the coordinates of the top-left x,
    # top_left y, right_bottom x, right_bottom y
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = 0

    for i in range(batch_size):
        # Get images form batch i
        img_pred = prediction[i]

        max_conf, max_conf_score = torch.max(img_pred[:, 5:5 + class_num], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (img_pred[:, :5], max_conf, max_conf_score)
        img_pred = torch.cat(seq, 1)

        non_zero_id = torch.nonzero(img_pred[:, 4])
        try:
            img_pred_ = img_pred[non_zero_id.squeeze(), :].view(-1, 7)
        except:
            continue

        if img_pred_.shape[0] == 0:
            continue

        # Class number of the images in the batch
        img_classes = unique(img_pred_[:, -1])

        for cls in img_classes:
            # Get a particular class
            cls_mask = img_pred_ * (img_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
            img_pred_class = img_pred_[class_mask_idx].view(-1, 7)

            # Sort detections so the maximum is at the top
            conf_sort_idx = torch.sort(img_pred_class[:, 4], descending=True)[1]
            img_pred_class = img_pred_class[conf_sort_idx]
            idx = img_pred_class.size(0)

            # Perform NMS
            for ind in range(idx):
                # Get all IOUS for boxes
                try:
                    IOUs = bbox_IOU(img_pred_class[ind].unsqueeze(0),
                                    img_pred_class[ind + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Remove b.boxes when iou < nms_conf
                IOU_mask = (IOUs < nms_conf).float().unsqueeze(1)
                img_pred_class[ind + 1:] *= IOU_mask
                non_zero_idx = torch.nonzero(img_pred_class[:, 4]).squeeze()
                img_pred_class = img_pred_class[non_zero_idx].view(-1, 7)

            # Repeat the batch_id for as many detections of the class cls in the
            # image
            batch_ind = img_pred_class.new(img_pred_class.size(0), 1).fill_(
                i)
            seq = batch_ind, img_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True

            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        # in case that output is empty
        return output

    except:
        return 0


def unique(tensor):
    """
        Args:
             tensor    : (tensor) input tensor
        Returns:
             Tensor used the method numpy.unique()
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_IOU(box1, box2, x1y1x2y2=True):
    '''
        Args:
             box1     : (tensor) coordinates of box1
             box2     : (tensor) coordinates of box2
        Returns:
             The IOU between box1 and box2
    '''
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def load_classes(classfile):
    '''
        Args:
             classfile   : (string) directory to class name file
        Returns:
             Splited file name list
    '''
    file = open(classfile, 'r')
    names = file.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    '''
        Args:
             img        : (numpy.array) input image
             inp_dim    : (list) required input image dimension
        Returns:
             Resized and padded image
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Created a canvas for padding
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] \
        = resized_img

    return canvas


def prep_image(img, inp_dim):
    '''
        Args:
             img        : (numpy.array) not pre-processed image
             inp_dim    : (list) required input image dimension
        Returns:
             Pre-processed images
    '''
    img = (letterbox_image(img, (inp_dim, inp_dim)))

    # Transform from BGR to RGB, HWC to CHW
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def pred_transform_train(model, prediction, dim_in, anchors, class_num, CUDA, target
                         , threshold, lamda_coord):
    '''
        Args:
             prediction     : (tensor) output from preceed layer
             dim_in         : (int) dimension of feature map
             anchor         : (list) anchor coordinates
             class_num      : (int) class numbers
             target         : (tensor) target labels
             threshold      : (float) threshold for objectness score
        Returns:
             Loss and predictions
    '''
    batch_size = prediction.size(0)
    stride = dim_in // prediction.size(2)
    grid_size = prediction.size(2)
    bbox_attr = 5 + class_num
    anchor_num = len(anchors)

    # Transfroms to the prediction
    prediction = prediction.view(batch_size, bbox_attr * anchor_num,
                                 grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, anchor_num, grid_size, grid_size,
                                 bbox_attr)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Adding sigmoid to the x_coord, y__coord and objscore
    prediction[..., 0] = torch.sigmoid(prediction[..., 0])
    prediction[..., 1] = torch.sigmoid(prediction[..., 1])
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])

    # Get outputs
    x = prediction[..., 0].clone()
    y = prediction[..., 1].clone()
    w = prediction[..., 2].clone()
    h = prediction[..., 3].clone()

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    # Add offset to the central coordinates
    offset_x = torch.FloatTensor(a).view(-1, 1)
    offset_y = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        offset_x = offset_x.cuda()
        offset_y = offset_y.cuda()

    offset_x_y = torch.cat((offset_x, offset_y), 1).view(-1, grid_size, grid_size,
                                                         2).repeat(1, 3, 1, 1, 1)
    prediction[..., :2] += offset_x_y

    # Add log-space transforms
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    scaled_anchors = anchors.repeat(batch_size, grid_size, grid_size, 1, 1) \
        .transpose(1, 3)
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * scaled_anchors

    # Add sigmoid to classes possibility
    prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])

    # Get parameters for loss evaluation
    pred_boxes = torch.FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = prediction[..., 0]
    pred_boxes[..., 1] = prediction[..., 1]
    pred_boxes[..., 2] = prediction[..., 2]
    pred_boxes[..., 3] = prediction[..., 3]
    conf = prediction[..., 4]
    pred_cls = prediction[..., 5:]

    # Prepare the labels
    nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_target(
        pred_boxes, target, anchors, class_num, grid_size, threshold)

    # Evaluate losses
    mask = Variable(torch.FloatTensor(mask))
    cls_mask = Variable(torch.FloatTensor(mask.unsqueeze(-1).repeat(1, 1, 1, 1, class_num)))
    conf_mask = Variable(torch.FloatTensor(conf_mask))

    tx = Variable(torch.FloatTensor(tx), requires_grad=False)
    ty = Variable(torch.FloatTensor(ty), requires_grad=False)
    tw = Variable(torch.FloatTensor(tw), requires_grad=False)
    th = Variable(torch.FloatTensor(th), requires_grad=False)
    tconf = Variable(torch.FloatTensor(tconf), requires_grad=False)
    tcls = Variable(torch.FloatTensor(tcls), requires_grad=False)

    mask = mask.cuda()
    conf_mask = conf_mask.cuda()
    cls_mask = cls_mask.cuda()
    tx = tx.cuda()
    ty = ty.cuda()
    tw = tw.cuda()
    th = th.cuda()
    tconf = tconf.cuda()
    tcls = tcls.cuda()

    loss_x = lamda_coord * model.bce_loss(x * mask, tx * mask)
    loss_y = lamda_coord * model.bce_loss(y * mask, ty * mask)
    loss_w = lamda_coord * model.mse_loss(w * mask, tw * mask) / 2
    loss_h = lamda_coord * model.mse_loss(h * mask, th * mask) / 2
    loss_conf = model.bce_loss(conf * conf_mask, tconf * conf_mask)
    loss_cls = model.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    # Resize the detection map to the original image size
    prediction[:, :, :4] *= stride
    prediction = prediction.view(batch_size, anchor_num * grid_size * grid_size,
                                 bbox_attr)

    # Get recall

    recall = float(nCorrect / nGT) if nGT else 1
    return prediction, loss, loss_x.item(), loss_y.item(), loss_w.item(), \
           loss_h.item(), loss_conf.item(), loss_cls.item(), recall


def build_target(pred_boxes, target, anchors, num_anchors, num_classes,
                 dim, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    dim = dim

    mask = torch.zeros(nB, nA, dim, dim)
    conf_mask = torch.ones(nB, nA, dim, dim)
    tx = torch.zeros(nB, nA, dim, dim)
    ty = torch.zeros(nB, nA, dim, dim)
    tw = torch.zeros(nB, nA, dim, dim)
    th = torch.zeros(nB, nA, dim, dim)
    tconf = torch.zeros(nB, nA, dim, dim)
    tcls = torch.zeros(nB, nA, dim, dim, num_classes)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * dim
            gy = target[b, t, 2] * dim
            gw = target[b, t, 3] * dim
            gh = target[b, t, 4] * dim
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)

            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors),
                                                                        2)), np.array(anchors)), 1))

            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_IOU(gt_box, anchor_shapes)

            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres] = 0

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)

            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj

            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_IOU(gt_box, pred_box, x1y1x2y2=False)
            tconf[b, best_n, gj, gi] = 1

            if iou > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls

def calcul_heatmap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap

def draw(img, coor, thick):
    '''
        Args:
            window_name: (string)
            img: (ndarray) image for annotating
            coor: (list or tuple) shape (16, 2)
            thick: (int) thick of the line
            key: (int) the length of time the window stays
    '''
    if not ((coor[0][0] == 0 and coor[0][1] == 0) or (coor[1][0] == 0 and coor[1][1] == 0)):
        img = cv2.line(img, coor[0], coor[1], (181, 102, 60), thick)
    if not ((coor[1][0] == 0 and coor[1][1] == 0) or (coor[2][0] == 0 and coor[2][1] == 0)):
        img = cv2.line(img, coor[1], coor[2], (250, 203, 91), thick)
    if not ((coor[2][0] == 0 and coor[2][1] == 0) or (coor[6][0] == 0 and coor[6][1] == 0)):
        img = cv2.line(img, coor[2], coor[6], (35, 98, 177), thick)
    if not ((coor[3][0] == 0 and coor[3][1] == 0) or (coor[6][0] == 0 and coor[6][1] == 0)):
        img = cv2.line(img, coor[3], coor[6], (35, 98, 177), thick)
    if not ((coor[3][0] == 0 and coor[3][1] == 0) or (coor[4][0] == 0 and coor[4][1] == 0)):
        img = cv2.line(img, coor[3], coor[4], (66, 218, 128), thick)
    if not ((coor[4][0] == 0 and coor[4][1] == 0) or (coor[5][0] == 0 and coor[5][1] == 0)):
        img = cv2.line(img, coor[4], coor[5], (62, 121, 58), thick)
    if not ((coor[6][0] == 0 and coor[6][1] == 0) or (coor[7][0] == 0 and coor[7][1] == 0)):
        img = cv2.line(img, coor[6], coor[7], (23, 25, 118), thick)
    if not ((coor[7][0] == 0 and coor[7][1] == 0) or (coor[8][0] == 0 and coor[8][1] == 0)):
        img = cv2.line(img, coor[7], coor[8], (152, 59, 98), thick)
    if not ((coor[8][0] == 0 and coor[8][1] == 0) or (coor[9][0] == 0 and coor[9][1] == 0)):
        img = cv2.line(img, coor[8], coor[9], (244, 60, 166), thick)
    if not ((coor[8][0] == 0 and coor[8][1] == 0) or (coor[12][0] == 0 and coor[12][1] == 0)):
        img = cv2.line(img, coor[8], coor[12], (244, 59, 166), thick)
    if not ((coor[11][0] == 0 and coor[11][1] == 0) or (coor[12][0] == 0 and coor[12][1] == 0)):
        img = cv2.line(img, coor[11], coor[12], (51, 135, 239), thick)
    if not ((coor[10][0] == 0 and coor[10][1] == 0) or (coor[11][0] == 0 and coor[11][1] == 0)):
        img = cv2.line(img, coor[10], coor[11], (35, 98, 177), thick)
    if not ((coor[8][0] == 0 and coor[8][1] == 0) or (coor[13][0] == 0 and coor[13][1] == 0)):
        img = cv2.line(img, coor[8], coor[13], (244, 59, 166), thick)
    if not ((coor[13][0] == 0 and coor[13][1] == 0) or (coor[14][0] == 0 and coor[14][1] == 0)):
        img = cv2.line(img, coor[13], coor[14], (49, 56, 218), thick)
    if not ((coor[14][0] == 0 and coor[14][1] == 0) or (coor[15][0] == 0 and coor[15][1] == 0)):
        img = cv2.line(img, coor[14], coor[15], (23, 25, 118), thick)
    return img


def dis(pt1, pt2):
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    return abs((dx ** 2 + dy ** 2) ** 0.5)

def insert_img(ori_img, ins_img, part, joint):
    parts = ['rank', 'rkne', 'rhip',
                  'lhip', 'lkne', 'lank',
                  'pelv', 'thrx', 'neck', 'head',
                  'rwri', 'relb', 'rsho',
                  'lsho', 'lelb', 'lwri']
    if isinstance(part, str):
        part = parts.index(part)
    point = list(joint[part])
    # Image has been sharpened is better
    white = np.array([242, 242, 242])

    # Using the size of head to confirm the size of insert-picture
    size = dis(joint[8], joint[9])
    # point[1] += int(size/9)

    # The original coordinate system is where the point is annotated.
    ori_height, ori_width = ori_img.shape[:2]
    ins_height, ins_width = ins_img.shape[:2]
    scale = abs(size / ins_height)
    if scale < 1:
        ins_img = cv2.resize(ins_img, (int(ins_width * scale), int(size)), interpolation=cv2.INTER_AREA)
    else:
        ins_img = cv2.resize(ins_img, (int(ins_width * scale), int(size)), interpolation=cv2.INTER_CUBIC)
    ins_height, ins_width = ins_img.shape[:2]
    x1, y1 = point[0] - ins_width // 2, point[1] - ins_height

    # It is impossible for y2 to get out of the original image
    ins_img = ins_img[max(-y1, 0):, max(-x1, 0):min(ins_width, ori_width - x1)]
    ins_height, ins_width = ins_img.shape[:2]
    for y in range(ins_height):
        for x in range(ins_width):
            # Insert pixel without white edge
            if not(ins_img[y][x] >= white).all():
                ori_img[y + max(y1, 0)][x + max(x1, 0)] = ins_img[y][x]

def get_points(heatmap):
    pt_np = np.zeros((16, 2), dtype=int)
    for part in range(16):
        part_target = heatmap[0, part + 16, :, :]
        if part_target.max() >= 0.4:
            pt_np[part][0] = np.where(part_target == part_target.max())[0][0]
            pt_np[part][1] = np.where(part_target == part_target.max())[1][0]
    pt = [[0, 0]] * 16
    for part in range(16):
        pt[part] = int(pt_np[part][0] * 4), int(pt_np[part][1] * 4)
    return pt

def get_points_multi(heatmap):
    '''
    Args:
        heatmap :(tensor) shape(batch_size, 32, 64, 64)
    Return  :
        pt_np   :(ndarray) shape(batch_size, 16, 2)
    '''
    num_pp = heatmap.shape[0]
    pt_np = np.zeros((num_pp, 16, 2), dtype=int)
    for person in range(num_pp):
        for part in range(16):
            part_target = heatmap[person, part + 16, :, :]
            if part_target.max() >= 0.05:
                pt_np[person][part][0] = np.where(part_target == part_target.max())[0][0]
                pt_np[person][part][1] = np.where(part_target == part_target.max())[1][0]
    return pt_np
