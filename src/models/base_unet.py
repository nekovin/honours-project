import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(BaseUNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),  # Keeps size same
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),  # Keeps size same
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Upsamples
        self.decoder4 = conv_block(1024, 512)  # Takes upsampled + encoder skip

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # Outputs single-channel denoised image

    def forward(self, x):
        enc1 = self.encoder1(x)  # (B, 64, H, W)
        enc2 = self.encoder2(self.pool(enc1))  # (B, 128, H/2, W/2)
        enc3 = self.encoder3(self.pool(enc2))  # (B, 256, H/4, W/4)
        enc4 = self.encoder4(self.pool(enc3))  # (B, 512, H/8, W/8)

        bottleneck = self.bottleneck(self.pool(enc4))  # (B, 1024, H/16, W/16)

        dec4 = self.upconv4(bottleneck)  # (B, 512, H/8, W/8)
        dec4 = torch.cat((dec4, enc4), dim=1)  # (B, 1024, H/8, W/8)
        dec4 = self.decoder4(dec4)  # (B, 512, H/8, W/8)

        dec3 = self.upconv3(dec4)  # (B, 256, H/4, W/4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # (B, 512, H/4, W/4)
        dec3 = self.decoder3(dec3)  # (B, 256, H/4, W/4)

        dec2 = self.upconv2(dec3)  # (B, 128, H/2, W/2)
        dec2 = torch.cat((dec2, enc2), dim=1)  # (B, 256, H/2, W/2)
        dec2 = self.decoder2(dec2)  # (B, 128, H/2, W/2)

        dec1 = self.upconv1(dec2)  # (B, 64, H, W)
        dec1 = torch.cat((dec1, enc1), dim=1)  # (B, 128, H, W)
        dec1 = self.decoder1(dec1)  # (B, 64, H, W)

        return self.final(dec1)  # (B, 1, H, W)
    
    def __str__(self):
        return "BaseUNet"


def get_base_unet_model(in_channels = 1, out_channels = 1, device='cuda'):
    model = BaseUNet(in_channels, out_channels)
    model.to(device)
    return model