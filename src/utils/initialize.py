# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 3 23:26 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Nov 3 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''
import torch.nn as nn

def initialize_weight(net):
    '''
    Args:
         net     : (nn.Module) network to be initialized
    '''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()



