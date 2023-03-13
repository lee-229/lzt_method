import torch
import torch.nn as nn
import math
from models.model import *
from models.swit import RSTB
image_size=128
#my_model_3_6
class my_model_3_6(nn.Module):#其实差不多 跟unet
    def __init__(self):
        super(my_model_3_6, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=2,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=2,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            nn.Conv2d(in_channels=128,
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


    ## num_heads=8 window_size=4
    def __init__(self):
        super(my_model_3_9, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            RSTB(dim=32,
                 input_resolution=(image_size, image_size),
                 # input_resolution=(1024, 1024),
                 depth=1,
                 num_heads=8,
                 window_size=4,
                 mlp_ratio=1,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,  # no impact on SR results
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=image_size,
                 patch_size=4,
                 resi_connection='1conv'),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            RSTB(dim=32,
                 input_resolution=(image_size, image_size),
                 # input_resolution=(1024, 1024),
                 depth=1,
                 num_heads=8,
                 window_size=4,
                 mlp_ratio=1,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,  # no impact on SR results
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=image_size,
                 patch_size=4,
                 resi_connection='1conv'),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            nn.Conv2d(in_channels=128,
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
class my_model_3_6_resnet(nn.Module):#其实差不多 跟unet
    #在3_6的基础上加入res结构
    def __init__(self):
        super(my_model_3_6_resnet, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=2,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=2,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            nn.Conv2d(in_channels=128,
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

    #在3_6_change的基础上加入res结构
    def __init__(self):
        super(my_model_3_6_change_resnet, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=4, head_dim=4,
                       window_size=8, relative_pos_embedding=True, cross_attn=False),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=4, head_dim=4,
                       window_size=8, relative_pos_embedding=True, cross_attn=False),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            nn.Conv2d(in_channels=128,
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
class my_model_3_6_resnet_noTrans(nn.Module):#其实差不多 跟unet
    #在3_6的基础上加入res结构
    def __init__(self):
        super(my_model_3_6_resnet_noTrans, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
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
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU())
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            nn.Conv2d(in_channels=128,
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

class my_model_3_10(nn.Module):#其实差不多 跟unet
    #my_model_3_6_resnet的基础上加入保持模块
    def __init__(self):
        super(my_model_3_10, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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

            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)

class my_model_3_11(nn.Module):#其实差不多 跟unet
    #在
    def __init__(self):
        super(my_model_3_11, self).__init__()
        self.encoder1_pan=nn.Sequential(
            SwinModule(in_channels=1, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            SwinModule(in_channels=4, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            nn.Conv2d(in_channels=128,
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

class my_model_3_11_2(nn.Module):#其实差不多 跟unet
    #最精简版的TFnet 
    def __init__(self):
        super(my_model_3_11_2, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
             nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
            # MultiAttentionResBlock(128, 128, 5)
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            #MultiAttentionResBlock(256, 256, 3),
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)

   
class my_model_3_11_3(nn.Module):#其实差不多 跟unet
    #最精简版的TFnet +transformer
    def __init__(self):
        super(my_model_3_11_3, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
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
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),)
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
            # MultiAttentionResBlock(128, 128, 5)
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            #MultiAttentionResBlock(256, 256, 3),
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)
class my_model_3_11_4(nn.Module):#其实差不多 跟unet
    #最精简版的TFnet +multiblock
    def __init__(self):
        super(my_model_3_11_4, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            # SwinModule(in_channels=32, hidden_dimension=32, layers=1,
            #            downscaling_factor=1, num_heads=8, head_dim=4,
            #            window_size=4, relative_pos_embedding=True, cross_attn=False),
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
            # SwinModule(in_channels=32, hidden_dimension=32, layers=1,
            #            downscaling_factor=1, num_heads=8, head_dim=4,
            #            window_size=4, relative_pos_embedding=True, cross_attn=False),
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)
class my_model_3_11_5(nn.Module):#其实差不多 跟unet
    #全用transformer
    def __init__(self):
        super(my_model_3_11_5, self).__init__()
        self.encoder1_pan=nn.Sequential(
            #  nn.Conv2d(in_channels=1,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            SwinModule(in_channels=1, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       )

        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            
        )
        self.encoder1_lr=nn.Sequential(
            #  nn.Conv2d(in_channels=4,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            SwinModule(in_channels=4, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
            #MultiAttentionResBlock(128, 128, 5)
            SwinModule(in_channels=128, hidden_dimension=128, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            SwinModule(in_channels=256, hidden_dimension=256, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)

class my_model_3_12(nn.Module):#其实差不多 跟unet
    #RSTB提取 不收敛
    #my_model_3_6_resnet的基础上加入保持模块
    def __init__(self):
        super(my_model_3_12, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            RSTB(input_dim=32,
                 embed_dim=32,
                 num_heads=4,
                 window_size=8,
                 patch_size=1,
                 resi_connection='1conv')
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            RSTB(input_dim=32,
                 embed_dim=32,
                 num_heads=4,
                 window_size=8,
                 patch_size=1,
                 resi_connection='1conv'))
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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

            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)

class my_model_3_12_2(nn.Module):#其实差不多 跟unet
    #最精简版的TFnet +multiblock+transformer+multiblock
    def __init__(self):
        super(my_model_3_12_2, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       
            MultiAttentionResBlock(32, 32, 5))

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
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
            MultiAttentionResBlock(32, 32, 5)
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)      
class my_model_3_12_3(nn.Module):#其实差不多 跟unet
    #最精简版的TFnet +multiblock+transformer+multiblock
    def __init__(self):
        super(my_model_3_12_3, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
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
            SwinModule(in_channels=32, hidden_dimension=32, layers=1,
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
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
            # nn.Conv2d(in_channels=128,
            #           out_channels=128,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
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

        return  torch.clamp(restore3+x_lr, -1, 1)      
      
   
         
   
   
   
#    
   
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = my_model_3_12().to(device)
# summary(model, ((1,1, 64,64),(1,4, 64,64)))