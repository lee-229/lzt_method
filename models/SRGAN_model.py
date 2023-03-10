import math
import torch
from torch import nn
# from torchinfo import summary
import torch.nn.functional as F
# class Generator(nn.Module):
#     def __init__(self, cfg,n_layers ,kw):
#         upsample_block_num = int(math.log(cfg.scale_factor, 2))
#
#         super(Generator, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#         self.block2 = ResidualBlock(64)
#         self.block3 = ResidualBlock(64)
#         self.block4 = ResidualBlock(64)
#         self.block5 = ResidualBlock(64)
#         self.block6 = ResidualBlock(64)
#         self.block7 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         #去掉上采样环节
#         # block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
#         self.block8 =nn.Conv2d(64, 4, kernel_size=9, padding=4)
#         self.gray = Gray(in_channel=4)
#
#         self.blur = kernel_spatial(n_layers,kw)
#
#         self.mse_loss = nn.MSELoss()
#         self.cfg = cfg
#
#     def forward(self,ms_lr_img ,ms_img, pan_img):
#         #input = torch.cat([ms_img, pan_img], dim=1)
#         input = ms_img
#         block1 = self.block1(input)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         block4 = self.block4(block3)
#         block5 = self.block5(block4)
#         block6 = self.block6(block5)
#         block7 = self.block7(block6)
#         block8 = self.block8(block1 + block7)
#         output = torch.tanh(block8)
#         loss_cfg = self.cfg.get('loss_cfg', {})
#
#         ms_img_gray = self.gray(ms_img)
#         pan_img_blur = self.blur(torch.cat([pan_img for _ in range(4)], dim=1))
#
#         output_blur = self.blur(output)
#         output_gray = self.gray(output)
#
#         fake_pan = torch.mean(output_gray, dim=1, keepdim=True)
#         fake_lr_up = output_blur
#
#         # reconstruction loss of generator
#         spatial_loss_rec = self.mse_loss(pan_img, fake_pan)
#         spectral_loss_rec = self.mse_loss(ms_lr_img, F.interpolate(output, scale_factor=0.25))
#
#         ms_img_gray = torch.mean(ms_img_gray, dim=1, keepdim=True)
#         pan_img_blur = torch.mean(pan_img_blur, dim=1, keepdim=True)
#         spatial_loss_RB = self.mse_loss(ms_img_gray, pan_img_blur)
#         spectral_loss_RB = self.mse_loss(fake_lr_up, ms_img)
#
#         G_loss = loss_cfg['spatial_loss_rec'].w * spatial_loss_rec + loss_cfg[
#                 'spectral_loss_rec'].w * spectral_loss_rec + \
#                      loss_cfg['spatial_loss_RB'].w * spatial_loss_RB + loss_cfg['spectral_loss_RB'].w * spectral_loss_RB
#
#         return output, fake_pan, fake_lr_up, G_loss


class Discriminator(nn.Module):
    def __init__(self,in_channel):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = Generator(scale_factor=2).to(device)
# summary(model, (1,4, 128,128))