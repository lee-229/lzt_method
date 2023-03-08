
import torch
import torch.nn as nn
import math
from torchinfo import summary
from function.utilis import high_pass,sobel_pass
from function.functions import *
# from function.loss import perception_loss,QNRLoss
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
class Gray(nn.Module):
    def __init__(self, in_channel=4, retio=4):
        super(Gray, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.in_channel = in_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel * retio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel * retio, in_channel, bias=False),
            nn.Sigmoid(),
            nn.Softmax()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        y = self.avg(x2).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = torch.sum(out, dim=1, keepdim=True)
        out = stack(out, r=self.in_channel)

        return out
class kernel_spatial(nn.Module):
    def __init__(self,n_layers,kw):
        super(kernel_spatial, self).__init__()
        layers=[]
        for n in range(1, n_layers):  # gradually increase the number of filters
            layers.append(nn.Conv2d(4,4, kernel_size=kw[n], stride=1, padding=kw[n]//2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x = self.conv1(x)
        output = self.net(x)
        return output
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
class generator(nn.Module):
    def __init__(self, cfg):
        super(generator, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ResidualBlock(32),
            ResidualBlock(32),

        )
        self.add =  self.layer3_pan = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.gray = Gray(in_channel=4)
        self.blur = kernel_spatial(2, [5,3])
        self.mse_loss = nn.L1Loss()
        self.percep_loss = perception_loss()
        self.QNR = QNRLoss()
        self.cfg = cfg
        # initialize_weights(self.extract, self.add, self.gray, self.blur)

    def forward(self, ms_img,ms_lr_img,pan_img):
        #利用GB和RB模块生成退化图像
        loss_cfg = self.cfg.get('loss_cfg', {})
        ms_img_gray = self.gray(ms_img)
        pan_img_blur = self.blur(torch.cat([pan_img for _ in range(4)], dim=1))
        #提取特征
        input = torch.cat([ms_img,pan_img],dim=1)
        out_128 = self.extract(input)
        # 注入多光谱图像
        out = self.add(out_128)+ms_img
        #对生成图像做退化 结果为fake_pan和fake_lr_up
        output_blur = self.blur(out)
        output_gray = self.gray(out)
        fake_pan = torch.mean(output_gray, dim=1, keepdim=True)
        fake_lr_up = output_blur

        # reconstruction loss of generator
        spatial_loss_high = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))+0.06*self.percep_loss(pan_img,fake_pan)
        #spatial_loss_high = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
        # loss of RB in low resolution
        ms_img_gray = torch.mean(ms_img_gray, dim=1, keepdim=True)
        pan_img_blur = torch.mean(pan_img_blur, dim=1, keepdim=True)
        spatial_loss_low = self.mse_loss(ms_img_gray, pan_img_blur)
        spectral_loss = self.mse_loss(F.interpolate(fake_lr_up, scale_factor=0.25), ms_lr_img)
        # QNR_loss = self.QNR(pan_img,ms_lr_img,out)
        G_loss_rec = loss_cfg['spatial_loss_high'].w * spatial_loss_high + loss_cfg[
            'spectral_loss'].w * spectral_loss +loss_cfg['spatial_loss_low'].w * spatial_loss_low\
        +loss_cfg['QNR_loss'].w * 0
        return out,fake_pan,G_loss_rec
class discriminator(nn.Module):
    #输入为n*c*128*128 输出为n*1
    def __init__(self,in_channel):
        super(discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Sigmoid()
            #nn.Linear(in_features=3 ** 2, out_features=1)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.layer6(x)
        return out

class discriminator_spec(nn.Module):
    #输入为n*c*32*32 输出为n*1
    def __init__(self,in_channel):
        super(discriminator_spec, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Sigmoid()
            #nn.Linear(in_features=3 ** 2, out_features=1)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.layer6(x)
        return out
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = discriminator_spec(in_channel=4).to(device)
# summary(model, (1,4, 32,32))