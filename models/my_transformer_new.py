import torch
import torch.nn as nn
import math
from models.model import *
def high_pass(img):
    device = img.device
    c = img.shape[1]
    img_hp = torch.zeros(img.shape).to(device)
    kernel = torch.tensor([[1.,1.,1.],
                          [1.,-8.,1.],
                          [1.,1.,1.]])
    kernel = kernel.expand((1, 1, 3, 3))
    kernel = kernel.to(device)
    for i in range(c):
        img_hp[:,i,:,:] = F.conv2d(img[:,i,:,:].unsqueeze(1), kernel, padding='same' , stride=1).squeeze(1)
    return img_hp
class UpSample(nn.Module):
    def __init__(self,  in_channels, scale_factor):
        super(UpSample, self).__init__()

        self.factor = scale_factor
        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))
        elif self.factor == 4:
            self.conv = nn.Conv2d(2*in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        

        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))

        return out
class my_model_4_4(nn.Module):#其实差不多 跟unet
    #在3_12的基础上 把swintransformer的MLP模块去掉
    def __init__(self):
        super(my_model_4_4, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule_change(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       
            MultiResBlock_noConv(32, 32, 5))

        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            
        )
        self.encoder1_lr=nn.Sequential(
             nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule_change(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
            MultiResBlock_noConv(32, 32, 5)
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            MultiAttentionResBlock(128, 128, 5)
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            MultiAttentionResBlock(256, 256, 3),
            UpSample(256,2),
            nn.PReLU())
        self.restore2=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            UpSample(128,2),
            nn.PReLU()
        )
        self.restore3=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            # nn.Conv2d(in_channels=64,
            #           out_channels=64,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return  torch.clamp(restore3, -1, 1)   
class my_model_4_4_2(nn.Module):#其实差不多 跟unet
    #在3_12的基础上 把swintransformer的MLP模块去掉
    def __init__(self):
        super(my_model_4_4_2, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
           
            )

        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
           
            
        )
        self.encoder1_lr=nn.Sequential(
             nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
           
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
 
            SwinModule_change(in_channels=128, hidden_dimension=128, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       
            MultiResBlock_noConv(128, 128, 5),
            SwinModule_change(in_channels=128, hidden_dimension=128, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       
            MultiResBlock_noConv(128, 128, 5),
            
            
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            
            SwinModule_change(in_channels=256, hidden_dimension=256, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False,shifted=True),
                       
            MultiResBlock_noConv(256, 256, 3),
            SwinModule_change(in_channels=256, hidden_dimension=256, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False,shifted=True),
                       
            MultiResBlock_noConv(256, 256, 3),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU())
        self.restore2=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
        self.restore3=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            
            nn.Conv2d(in_channels=64,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return  torch.clamp(restore3+x_lr, -1, 1)  
class my_model_4_4_3(nn.Module):#其实差不多 跟unet
    #在3_12的基础上 把swintransformer的MLP模块去掉
    def __init__(self):
        super(my_model_4_4_3, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
           
            )

        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
           
            
        )
        self.encoder1_lr=nn.Sequential(
             nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
           
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
 
            SwinModule_change(in_channels=128, hidden_dimension=128, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       
            MultiResBlock_noConv(128, 128, 5),
           
            
            
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            
            SwinModule_change(in_channels=256, hidden_dimension=256, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False,shifted=True),
                       
            MultiResBlock_noConv(256, 256, 3),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU())
        self.restore2=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
        self.restore3=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            
            nn.Conv2d(in_channels=64,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        x_lr_original=x_lr
        x_pan=high_pass(x_pan)
        x_lr =high_pass(x_lr)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return  torch.clamp(restore3+x_lr_original, -1, 1)  
      