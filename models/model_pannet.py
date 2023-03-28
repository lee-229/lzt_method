
import torch.nn as nn
import torch


import numpy as np
import cv2
import kornia

    
import torch

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        fea = self.relu(self.bn1(self.conv1(x)))
        fea = self.relu(self.bn2(self.conv2(fea)))
        result = fea + x
        return result
       

class PanNet_model(nn.Module):
    def __init__(self):
        super(PanNet_model, self).__init__()
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=8, stride=4, padding=2, output_padding=0)
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            self.make_layer(Residual_Block, 4, 32),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)
        )
        self.tan=nn.Tanh()

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, y, x):
        lr_up = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic')
        lr_hp = x - kornia.filters.BoxBlur((5, 5))(x)
        pan_hp = y - kornia.filters.BoxBlur((5, 5))(y)
        lr_u_hp = self.layer_0(lr_hp)#self.bicubic(lr_hp, scale=cfg.scale)#
        ms = torch.cat([pan_hp, lr_u_hp], dim=1)
        fea = self.layer_1(ms)
        output = self.layer_2(fea) + lr_up
        output=self.tan(output)
        return torch.clamp(output, -1, 1)  
class PanNet_model_8c(nn.Module):
    def __init__(self):
        super(PanNet_model_8c, self).__init__()
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=8, stride=4, padding=2, output_padding=0)
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            self.make_layer(Residual_Block, 4, 32),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1)
        )
        self.tan=nn.Tanh()

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, y, x):
        lr_up = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic')
        lr_hp = x - kornia.filters.BoxBlur((5, 5))(x)
        pan_hp = y - kornia.filters.BoxBlur((5, 5))(y)
        lr_u_hp = self.layer_0(lr_hp)#self.bicubic(lr_hp, scale=cfg.scale)#
        ms = torch.cat([pan_hp, lr_u_hp], dim=1)
        fea = self.layer_1(ms)
        output = self.layer_2(fea) + lr_up
        output=self.tan(output)
        return torch.clamp(output, -1, 1)  
