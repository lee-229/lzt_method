import math
import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from model import Gray,kernel_spatial
class Generator(nn.Module):
    """docstring for Generator"""

    def __init__(self):
        super(Generator, self).__init__()

        # self.Vis_DeCon = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 1, 1))
        # self.IR_DeCon = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 4, 4))

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU( inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU( inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU( inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU( inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU( inplace=True))

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(80, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU( inplace=True))

        self.Decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU( inplace=True))

        self.Decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU( inplace=True))

        self.Decoder4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU( inplace=True))

        self.Decoder5 = nn.Sequential(
            nn.Conv2d(16, 4, 3, 1, 1),
            nn.Tanh( ))
        # self.gray = Gray(in_channel=4)
        #
        # self.blur = kernel_spatial(n_layers, kw)
        #
        self.mse_loss = nn.MSELoss()
        # self.cfg = cfg

    def forward(self,ms_lr_img ,ms_img, pan_img):
        # vis = self.Vis_DeCon(vis)
        # ir = self.IR_DeCon(ir)
        # x = torch.cat((vis, ir), 1)
        input = torch.cat([ms_img, pan_img], dim=1)
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat((x1, x2), 1))
        x4 = self.conv4(torch.cat((x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x1, x2, x3, x4), 1))
        x = torch.cat((x1, x2, x3, x4, x5), 1)

        x = self.Decoder1(x)
        x = self.Decoder2(x)
        x = self.Decoder3(x)
        x = self.Decoder4(x)
        output = self.Decoder5(x)
        # loss_cfg = self.cfg.get('loss_cfg', {})
        #
        # ms_img_gray = self.gray(ms_img)
        # pan_img_blur = self.blur(torch.cat([pan_img for _ in range(4)], dim=1))
        #
        # output_blur = self.blur(output)
        # output_gray = self.gray(output)
        #
        fake_pan = torch.mean(output, dim=1, keepdim=True)
        # fake_lr_up = output_blur
        #
        # # reconstruction loss of generator
        spatial_loss_rec = self.mse_loss(pan_img, fake_pan)
        spectral_loss_rec = self.mse_loss(ms_lr_img, F.interpolate(output, scale_factor=0.25))
        #
        # ms_img_gray = torch.mean(ms_img_gray, dim=1, keepdim=True)
        # pan_img_blur = torch.mean(pan_img_blur, dim=1, keepdim=True)
        # spatial_loss_RB = self.mse_loss(ms_img_gray, pan_img_blur)
        # spectral_loss_RB = self.mse_loss(fake_lr_up, ms_img)
        #
        G_loss = 5 * spatial_loss_rec + 5*spectral_loss_rec
        # + \
        #          loss_cfg['spatial_loss_RB'].w * spatial_loss_RB + loss_cfg['spectral_loss_RB'].w * spectral_loss_RB

        return output, fake_pan,  G_loss
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = Generator().to(device)
# summary(model, (1,5, 128,128))