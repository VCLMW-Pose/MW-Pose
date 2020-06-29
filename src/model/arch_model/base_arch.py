from .vgg import VGGBackbone
import torch.nn as nn
import torch


class BaseArch(nn.Module):
    def __init__(self, backbone: str):
        super(BaseArch, self).__init__()
        self.backbone = backbone

        if self.backbone == "VGG":
            self.encoder_block1 = VGGBackbone(59, 64, 2)
            self.encoder_ds1 = nn.MaxPool2d(2)
            self.encoder_block2 = VGGBackbone(64, 128, 2)
            self.encoder_ds2 = nn.MaxPool2d(2)
            self.encoder_block3 = VGGBackbone(128, 256, 3, 256)
            self.encoder_ds3 = nn.MaxPool2d(2)
            self.encoder_block4 = VGGBackbone(256, 512, 3, 512)
            self.encoder_ds4 = nn.MaxPool2d(2)
            self.encoder_block5 = VGGBackbone(512, 512, 3, 512)

            self.FC1 = nn.Linear(4608, 4608)
            self.FC2 = nn.Linear(4608, 4608)

            self.decoder_block1 = VGGBackbone(512, 512, 3, 512)
            self.decoder_block2 = VGGBackbone(1024, 256, 3, 512)
            self.decoder_block3 = VGGBackbone(512, 128, 3, 256)
            self.decoder_block4 = VGGBackbone(256, 64, 2)
            self.decoder_block5 = VGGBackbone(128, 32, 2)
            self.decoder_block6 = VGGBackbone(32, 16, 2)
            self.decoder_block7 = VGGBackbone(16, 14, 2)

    def forward(self, x):
        if self.backbone == "VGG":
            x = self.encoder_block1(x)
            shortcut1 = x.clone()

            x = self.encoder_ds1(x)
            x = self.encoder_block2(x)
            shortcut2 = x.clone()

            x = self.encoder_ds2(x)
            x = self.encoder_block3(x)
            shortcut3 = x.clone()

            x = self.encoder_ds3(x)
            x = self.encoder_block4(x)
            shortcut4 = x.clone()

            x = self.encoder_ds4(x)
            x = self.encoder_block5(x)

            x = x.view(x.size(0), -1)
            x = self.FC1(x)
            x = self.FC2(x)

            x = x.view([x.size(0), 512, 3, 3])
            x = self.decoder_block1(x)

            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)
            x = torch.cat((x, shortcut4), 1)
            x = self.decoder_block2(x)

            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)
            x = torch.cat((x, shortcut3), 1)
            x = self.decoder_block3(x)

            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)
            x = torch.cat((x, shortcut2), 1)
            x = self.decoder_block4(x)

            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)
            x = torch.cat((x, shortcut1), 1)
            x = self.decoder_block5(x)

            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)
            x = self.decoder_block6(x)

            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)
            x = self.decoder_block7(x)

            return x

