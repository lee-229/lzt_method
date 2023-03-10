import torch
import torch.nn as nn
import math
from torchinfo import summary
from function.utilis import high_pass,sobel_pass
from function.functions import *
#from function.loss import perception_loss
#from models.model_LDPNet import Gray,kernel_spatial
class generator(nn.Module):
    def __init__(self,in_size=128):
        super(generator, self).__init__()
        self.layer1_ms = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.layer1_pan = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.block = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            self.block,
            self.block,
            self.block,
            self.block
        )
        self.layer3 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)

        self.mse_loss = nn.MSELoss()

    def forward(self, ms_img, pan_img):
        x1 = torch.cat([self.layer1_ms(ms_img), self.layer1_pan(pan_img)], dim=1)
        out = self.layer3(self.layer2(x1))
        fake_pan = torch.mean(out+ms_img, dim=1, keepdim=True)
        # reconstruction loss of generator
        spatial_loss_rec = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
        spectral_loss_rec = self.mse_loss(out+ms_img, ms_img)
        G_loss_rec = 20*spatial_loss_rec+spectral_loss_rec
        return out+ms_img,fake_pan,G_loss_rec

class discriminator(nn.Module):
    #输入为n*c*128*128 输出为n*1
    def __init__(self,in_channel):
        super(discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128 ,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid() ##删掉即为WGAN


        )



    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layer5(x)

        return out
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = discriminator(in_channel=4).to(device)
# summary(model, (1,4, 128,128))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        # residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        # residual = self.bn2(residual)

        return x + residual
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=32, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))#这里指通道数的叠加，所以每次叠加结束，通道数都增加num_grow_ch
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5*0.2+x
class BDPN(nn.Module):
    def __init__(self, in_size=128):
        super(BDPN, self).__init__()
        self.layer1_pan = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0),

        )
        # 用于降维
        self.layer2_pan = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)
        )
        self.layer1_ms = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2)
        )
        # 用于降维
        self.layer2_ms = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2)
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, ms_up_img,ms_img,pan_img):
        pan_out_1 = self.layer1_pan(pan_img)
        pan_out_2 = self.layer2_pan(pan_out_1)

        ms_out_1 = self.layer1_ms(ms_img)+pan_out_2
        ms_out_2 = self.layer2_ms(ms_out_1)+pan_out_1

        fake_pan = torch.mean(ms_out_2, dim=1, keepdim=True)
        # reconstruction loss of generator
        spatial_loss_rec = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
        spectral_loss_rec = self.mse_loss(F.interpolate(ms_out_2, scale_factor=0.25), ms_img)
        G_loss_rec = 20 * spatial_loss_rec + spectral_loss_rec
        return ms_out_2,fake_pan,G_loss_rec
# class P3Net(nn.Module):
#     def __init__(self, in_size=128):
#         super(P3Net, self).__init__()
#         self.extract = nn.Sequential(
#             nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             ResidualBlock(32),
#             ResidualBlock(32),
#
#         )
#         self.add =  self.layer3_pan = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
#             nn.Tanh()
#         )
#         self.gray = Gray(in_channel=4)
#
#         self.blur = kernel_spatial(2, [5,3])
#         self.mse_loss = nn.L1Loss()
#         self.percep_loss = perception_loss()
#
#     def forward(self, ms_img,ms_lr_img,pan_img):
#         #利用GB和RB模块生成退化图像
#         ms_img_gray = self.gray(ms_img)
#         pan_img_blur = self.blur(torch.cat([pan_img for _ in range(4)], dim=1))
#         #提取特征
#         input = torch.cat([ms_img,pan_img],dim=1)
#         out_128 = self.extract(input)
#         # out_64 =self.extract(F.interpolate(input, scale_factor=0.5))
#         # out_32 = self.extract(F.interpolate(input, scale_factor=0.25))
#         # out = self.upscale(self.upscale(out_32)+out_64)+out_128
#         # 注入多光谱图像
#         out = self.add(out_128)+ms_img
#         #对生成图像做退化 结果为fake_pan和fake_lr_up
#         output_blur = self.blur(out)
#         output_gray = self.gray(out)
#         fake_pan = torch.mean(output_gray, dim=1, keepdim=True)
#         fake_lr_up = output_blur
#
#         # reconstruction loss of generator
#         spatial_loss_high = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))+0.06*self.percep_loss(pan_img,fake_pan) #self.mse_loss(high_pass(pan_img), high_pass(fake_pan)) +
#         # loss of RB in low resolution
#         ms_img_gray = torch.mean(ms_img_gray, dim=1, keepdim=True)
#         pan_img_blur = torch.mean(pan_img_blur, dim=1, keepdim=True)
#         spatial_loss_low = self.mse_loss(ms_img_gray, pan_img_blur)
#
#         spectral_loss = self.mse_loss(F.interpolate(fake_lr_up, scale_factor=0.25), ms_lr_img)
#         G_loss_rec = 5 * spatial_loss_high + spectral_loss + 0.5*spatial_loss_low
#         return out,fake_pan,G_loss_rec

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = discriminator(4).to(device)
# summary(model, (1,4, 128,128))


