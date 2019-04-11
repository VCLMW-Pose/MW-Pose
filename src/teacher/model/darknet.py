# -*- coding: utf-8 -*-
'''
    Created on wed Sept 5 21:12 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 25 23:28 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
How to implement a YOLO (v3) object detector from strach in
PyTorch: Part 2
'''

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_test_input(imgfile):
    '''
        Args:
             imgfile   : (string) directory to image file
        Returns:
             Pre-processed image tensor
    '''
    img = cv2.imread(imgfile)
    img = cv2.resize(img, (416,416))
    img = img[:, :, ::-1].transpose((2, 0, 1))

    # Add a dimension for batch & normalize
    img = img[np.newaxis, :, :]/255.0
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img

def parse_model_config(path):
    '''
        Args:
             path      : (string) directory to the *.cfg file
        Returns:
             Module definitions
    '''
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

# EmptyLayer is registered as the route layer in darknet
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# DetectLayer is registered for the yolo layer in darknet
class DetectLayer(nn.Module):
    def __init__(self, anchors):
        '''
            Args:
                 anchors   : (list) the list describing anchors of
                             yolo v3
        '''
        super(DetectLayer, self).__init__()
        self.anchors = anchors

class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim):
        '''
            Args:
                 anchors     : (list) the list describing anchors
                 num_classes : (int) number of class
                 img_dim     : (int) image dimension
        '''
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, targets=None):
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  self.img_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim,
                                g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(
                    bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(
                    bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = \
                build_target(pred_boxes.cpu().data, targets.cpu().data, scaled_anchors,
                self.num_anchors, self.num_classes, g_dim, self.ignore_thres, self.img_dim)

            nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = Variable(mask.type(FloatTensor))
            cls_mask = Variable(mask.unsqueeze(-1).repeat(1, 1, 1, 1,
                                                self.num_classes).type(FloatTensor))
            conf_mask = Variable(conf_mask.type(FloatTensor))

            # Handle target variables
            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls.type(FloatTensor), requires_grad=False)

            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask)/2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask)/2
            loss_conf = self.bce_loss(conf*conf_mask, tconf*conf_mask)
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item()\
                                        , loss_conf.item(), loss_cls.item(), recall

        else:
            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(
                        bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

def create_modules(module_defs):
    """
        Args:
             module_defs   : (list) list of dictionary of module definition
        Returns:
             Built module list and hyper parameters
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample( scale_factor=int(module_def['stride']),
                                    mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]

            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])

            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

# darknet structure definition
class darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, class_num, img_size=416):
        '''
            Args:
                 config_path      : (string) directory to the cfg file
                 img_size         : (int) image input dimension
        '''
        super(darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.class_num = class_num
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = {'x': 0, 'y': 0, 'w': 0, 'h': 0, 'conf': 0, 'cls': 0, 'recall': 0}
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss

                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses['recall'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weight(self, weightfile):
        '''
            Args:
                 weightfile   : (string) directory to the weight file
        '''
        #Open the weights file

        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values

        # Needed to write header when saving weights
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':

                conv_layer = module[0]

                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # Number of biases

                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b

                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b

                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b

                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weight(self, filename):
        '''
            Args:
                 dir          : (string) directory of the destination
                 filename     : (string) file name of the saved model
            Returns:
                 Save weights for the model
        '''
        fp = open(filename, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(
                self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':

                conv_layer = module[0]

                 # If batch norm, load bn first
                if module_def['batch_normalize']:

                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                # Load conv bias
                else:

                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                    # Load conv weights

                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()




