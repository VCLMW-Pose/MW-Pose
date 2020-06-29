import torch.nn as nn


class ResNetBackbone(nn.Module):
    def __init__(self, chanin, chanout):