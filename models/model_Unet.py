import torch
import torch.nn as nn
import math
from torchinfo import summary
from models.model import *
#from function.utilis import high_pass
#from main.test import ConvMultiPatchAttention
from models.swit import RSTB
#image_size=128
class Unet_transformer_best(nn.Module):
    def __init__(self):
        super(Unet_transformer_best, self).__init__()
        self.encoder1_pan=nn.Sequential(

            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            # nn.Conv2d(in_channels=32,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()

        )
        self.encoder2_pan = nn.Sequential(

            RSTB(dim=32,
                 input_resolution=(image_size,image_size),
                 # input_resolution=(1024, 1024),
                 depth=1,
                 num_heads=4,
                 window_size=8,
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
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()

        )
        self.encoder3_pan = nn.Sequential(
            RSTB(dim=64,
                 input_resolution= (image_size//2,image_size//2),
                 depth=1,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=1,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,  # no impact on SR results
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=image_size//2,
                 patch_size=4,
                 resi_connection='1conv'),
            nn.Conv2d(in_channels=64,
                  out_channels=128,
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
            # nn.Conv2d(in_channels=32,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()

        )
        self.encoder2_lr = nn.Sequential(

            RSTB(dim=32,
                 input_resolution= (image_size,image_size),
                 depth=1,
                 num_heads=4,
                 window_size=8,
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
            nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=2,
                  stride=2),
            nn.PReLU()
        )
        self.encoder3_lr = nn.Sequential(
            RSTB(dim=64,
                 input_resolution= (image_size//2,image_size//2),
                 depth=1,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=1,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,  # no impact on SR results
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=image_size//2,
                 patch_size=4,
                 resi_connection='1conv'),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            MultiAttentionResBlock(256, 256, 5)
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
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
            nn.PReLU())

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
        # x_pan=torch.cat((x_pan,x_pan,x_pan,x_pan),dim=1)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        encoder3_pan = self.encoder3_pan(encoder2_pan)
        encoder3_lr = self.encoder3_lr(encoder2_lr)

        fusion1 = self.fusion1(torch.cat((encoder3_pan, encoder3_lr), dim=1))
        restore1 = self.restore1(fusion1)
        #restore1 = self.restore1(torch.cat((fusion1,encoder3_pan, encoder3_lr),dim=1))
        restore2 = self.restore2(torch.cat((restore1, encoder2_pan,encoder2_lr),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3
class _Residual_Block(nn.Module):
    def __init__(self, channels):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        #self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        #self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity_data = x
        output = self.prelu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        output = self.prelu(output)
        return output

class Unet_cutblock(nn.Module):
    def __init__(self):
        super(Unet_cutblock, self).__init__()
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
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder3_pan = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
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
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder3_lr = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            # nn.Conv2d(in_channels=256,
            #           out_channels=256,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU(),
            #MultiAttentionResBlock(256, 256, 5)
            ConvMultiPatchAttention(256)
            # nn.Conv2d(in_channels=256,
            #           out_channels=256,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
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
            nn.PReLU())

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
        # x_pan=torch.cat((x_pan,x_pan,x_pan,x_pan),dim=1)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        encoder3_pan = self.encoder3_pan(encoder2_pan)
        encoder3_lr = self.encoder3_lr(encoder2_lr)

        fusion1 = self.fusion1(torch.cat((encoder3_pan, encoder3_lr), dim=1))
        restore1 = self.restore1(fusion1)
        #restore1 = self.restore1(torch.cat((fusion1,encoder3_pan, encoder3_lr),dim=1))
        restore2 = self.restore2(torch.cat((restore1, encoder2_pan,encoder2_lr),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
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
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder3_pan = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
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
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder3_lr = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            MultiAttentionResBlock(256, 256, 5)
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
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
            nn.PReLU())

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
        # x_pan=torch.cat((x_pan,x_pan,x_pan,x_pan),dim=1)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        encoder3_pan = self.encoder3_pan(encoder2_pan)
        encoder3_lr = self.encoder3_lr(encoder2_lr)

        fusion1 = self.fusion1(torch.cat((encoder3_pan, encoder3_lr), dim=1))
        restore1 = self.restore1(fusion1)
        #restore1 = self.restore1(torch.cat((fusion1,encoder3_pan, encoder3_lr),dim=1))
        restore2 = self.restore2(torch.cat((restore1, encoder2_pan,encoder2_lr),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3
class Unet_transformer(nn.Module):
    def __init__(self):
        super(Unet_transformer, self).__init__()
        self.encoder1_pan=nn.Sequential(

            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            # nn.Conv2d(in_channels=32,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()

        )
        self.encoder2_pan = nn.Sequential(

            RSTB(dim=32,
                 input_resolution=(image_size,image_size),
                 # input_resolution=(1024, 1024),
                 depth=1,
                 num_heads=4,
                 window_size=8,
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
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()

        )
        self.encoder3_pan = nn.Sequential(
            RSTB(dim=64,
                 input_resolution= (image_size//2,image_size//2),
                 depth=1,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=1,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,  # no impact on SR results
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=image_size//2,
                 patch_size=4,
                 resi_connection='1conv'),
            nn.Conv2d(in_channels=64,
                  out_channels=128,
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
            # nn.Conv2d(in_channels=32,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()

        )
        self.encoder2_lr = nn.Sequential(

            RSTB(dim=32,
                 input_resolution= (image_size,image_size),
                 depth=1,
                 num_heads=4,
                 window_size=8,
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
            nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=2,
                  stride=2),
            nn.PReLU()
        )
        self.encoder3_lr = nn.Sequential(
            RSTB(dim=64,
                 input_resolution= (image_size//2,image_size//2),
                 depth=1,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=1,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,  # no impact on SR results
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=image_size//2,
                 patch_size=4,
                 resi_connection='1conv'),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            MultiAttentionResBlock(256, 256, 5)
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
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
            nn.PReLU())

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
        # x_pan=torch.cat((x_pan,x_pan,x_pan,x_pan),dim=1)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        encoder3_pan = self.encoder3_pan(encoder2_pan)
        encoder3_lr = self.encoder3_lr(encoder2_lr)

        fusion1 = self.fusion1(torch.cat((encoder3_pan, encoder3_lr), dim=1))
        restore1 = self.restore1(fusion1)
        #restore1 = self.restore1(torch.cat((fusion1,encoder3_pan, encoder3_lr),dim=1))
        restore2 = self.restore2(torch.cat((restore1, encoder2_pan,encoder2_lr),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3

class Unet_transformer_3_12(nn.Module):
    def __init__(self):
        super(Unet_transformer_3_12, self).__init__()
        self.encoder1_pan=nn.Sequential(

            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),

        )

        self.encoder1_lr=nn.Sequential(

            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            # nn.Conv2d(in_channels=32,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.PReLU()

        )
 
        self.fusion1=nn.Sequential(
            MultiAttentionResBlock(256, 256, 5)
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
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
            nn.PReLU())

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
        # x_pan=torch.cat((x_pan,x_pan,x_pan,x_pan),dim=1)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)
        _,_,image_size,image_size=encoder1_lr.shape
        self.encoder2_pan=RSTB(input_dim=32,
                embed_dim=64,
                image_size=image_size,
                 num_heads=4,
                 window_size=8,
                 patch_size=2,
                resi_connection='1conv').cuda()
        self.encoder2_lr=RSTB(input_dim=32,
                embed_dim=64,
                image_size=image_size,
                 num_heads=4,
                 window_size=8,
                 patch_size=2,
                resi_connection='1conv').cuda()
        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)
        _,_,image_size,image_size=encoder2_lr.shape
        self.encoder3_lr=RSTB(input_dim=64,
                embed_dim=128,
                image_size=image_size,
                 num_heads=4,
                 window_size=8,
                 patch_size=2).cuda()
        self.encoder3_pan=RSTB(input_dim=64,
                embed_dim=128,
                image_size=image_size,
                 num_heads=4,
                 window_size=8,
                 patch_size=2,
                resi_connection='1conv').cuda()
        encoder3_pan = self.encoder3_pan(encoder2_pan)
        encoder3_lr = self.encoder3_lr(encoder2_lr)

        fusion1 = self.fusion1(torch.cat((encoder3_pan, encoder3_lr), dim=1))
        restore1 = self.restore1(fusion1)
        #restore1 = self.restore1(torch.cat((fusion1,encoder3_pan, encoder3_lr),dim=1))
        restore2 = self.restore2(torch.cat((restore1, encoder2_pan,encoder2_lr),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3+x_lr






class Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.PReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.PReLU()
        )

    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1


class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.conv1 =nn.Sequential(
            DeepFeatureExtractBlock(5,64,3),
            # nn.Conv2d(5, 64, 3, stride=1, padding=1),
            # nn.PReLU()
        )

        self.residual_block = nn.Sequential(
            Block(),
            Block(),
            Block(),
        )

        self.restore=nn.Sequential(
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
        input = torch.cat((x_pan, x_lr), dim=1)
        x=self.conv1(input)
        x=self.residual_block(x)
        output=self.restore(x)

        return output




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = Unet_transformer_3_12().to(device)
# summary(model, ((1,1, 64,64),(1,4, 64,64)))