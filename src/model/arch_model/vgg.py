import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, chanin, chanout):
        super(Conv2d, self).__init__()
        self.ReLU = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(chanin, chanout, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.ReLU(x)
        return x


class VGGBackbone(nn.Module):
    def __init__(self, chanin: int, chanout: int, layer_num: int, chanmid=2):
        super(VGGBackbone, self).__init__()
        # assert(layer_num == 2 or layer_num == 3, "Layer number of VGG backbone can only be 2 or 3!")

        if layer_num == 2:
            self.layers = nn.Sequential(
                Conv2d(chanin, chanout),
                Conv2d(chanout, chanout)
            )
        else:
            self.layers = nn.Sequential(
                Conv2d(chanin, chanmid),
                Conv2d(chanmid, chanmid),
                Conv2d(chanmid, chanout)
            )

    def forward(self, x):
        x = self.layers(x)
        return x
