from function.functions import *
from function.utilis import high_pass
from torchinfo import summary
# device = torch.device('cuda' ) # PyTorch v0.4.0
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
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
            nn.Flatten(), #把n*n的图像展平为一列
            # nn.Sigmoid()
            nn.Linear(in_features=5 ** 2, out_features=9)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.layer6(x)
        return out

class discriminator2(nn.Module):
    #输入为n*c*32*32 输出为n*1
    def __init__(self,in_channel):
        super(discriminator2, self).__init__()
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
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            # nn.Sigmoid(),
            nn.Linear(in_features=3 ** 2, out_features=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.layer6(x)
        return out
class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class generator(nn.Module):
    #输入为n*5*128*128 输出为n*4*128*128
    def __init__(self,in_size=128):
        super(generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(64,momentum=0.9,eps=1e-5,affine=True),
            nn.ReLU( )
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64+5, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32,momentum=0.9,eps=1e-5,affine=True),
            nn.ReLU( )
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64+32+5, 4, kernel_size=5, stride=1, padding=2),
            nn.Tanh()#范围为0到1
        )


    def forward(self, ms_img, pan_img):

        input = torch.cat([ms_img, pan_img], dim=1)
        x1 = self.layer1(input)
        x2 = self.layer2(torch.cat([input, x1], dim=1))
        x3 = self.layer3(torch.cat([input, x1,x2], dim=1))
        return x3

# class generator2(nn.Module):
#     #输入为n*5*128*128 输出为n*4*128*128
#     def __init__(self,cfg,n_layers ,kw):
#         super(generator2, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(5, 64, kernel_size=9, stride=1, padding=4),
#             nn.BatchNorm2d(64,momentum=0.9,eps=1e-5,affine=True),
#             nn.PReLU( )
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64+5, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32,momentum=0.9,eps=1e-5,affine=True),
#             nn.PReLU( )
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64+32+5, 4, kernel_size=5, stride=1, padding=2),
#             nn.PReLU(),
#             nn.Tanh()#范围为0到1
#         )
#
#         self.gray = Gray(in_channel=4)
#
#         self.blur = kernel_spatial(n_layers,kw)
#
#         self.mse_loss = nn.MSELoss()
#         self.cfg = cfg
#
#     def forward(self,ms_lr_img ,ms_img, pan_img):
#         loss_cfg = self.cfg.get('loss_cfg', {})
#         input = torch.cat([ms_img, pan_img], dim=1)
#         ms_img_gray = self.gray(ms_img)
#         pan_img_blur = self.blur(torch.cat([pan_img for _ in range(4)], dim=1))
#         x1 = self.layer1(input)
#         x2 = self.layer2(torch.cat([input, x1], dim=1))
#         output = self.layer3(torch.cat([input, x1,x2], dim=1))
#         output_blur = self.blur(output)
#         output_gray = self.gray(output)
#
#         fake_pan = torch.mean(output_gray, dim=1, keepdim=True)
#         fake_lr_up = output_blur
#
#         # reconstruction loss of generator
#         spatial_loss_rec = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
#         spectral_loss_rec = self.mse_loss(ms_lr_img, F.interpolate(output, scale_factor=0.25))
#
#         ms_img_gray = torch.mean(ms_img_gray, dim=1, keepdim=True)
#         pan_img_blur = torch.mean(pan_img_blur, dim=1, keepdim=True)
#         spatial_loss_RB = self.mse_loss(high_pass(ms_img_gray), high_pass(pan_img_blur))
#         spectral_loss_RB = self.mse_loss(fake_lr_up, ms_img)
#
#         G_loss =  loss_cfg['spatial_loss_rec'].w * spatial_loss_rec + loss_cfg['spectral_loss_rec'].w * spectral_loss_rec + \
#                   loss_cfg['spatial_loss_RB'].w*spatial_loss_RB  + loss_cfg['spectral_loss_RB'].w*spectral_loss_RB
#
#         return output,fake_pan,fake_lr_up,G_loss
class Pangan(nn.Module):
    def __init__(self):
        super(Pangan, self).__init__()
        self.spectral_D = discriminator(4)
        self.spatial_D =discriminator(1)
        self.G = generator()
        initialize_weights(self.spectral_D, self.spatial_D, self.G)

    def forward(self, ms, pan):
        # fuse upsampled LRMS and PAN image 128*128*4 128*128*1
        output = self.G(ms, pan)
        return output


class Block(nn.Module):
    def __init__(self, input_channel=64, output_channel=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(),

            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel)
        )

    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1

# class Generator_SRGAN(nn.Module):
#     def __init__(self, cfg,n_layers ,kw):
#         """放大倍数是scale的平方倍"""
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(5, 64, 9, stride=1, padding=4),
#             nn.PReLU()
#         )
#         self.residual_block = nn.Sequential(
#             Block(),
#             Block(),
#             # Block(),
#             # Block(),
#             # Block(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             # nn.PixelShuffle(scale),
#             nn.PReLU(),
#             #
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             # nn.PixelShuffle(scale),
#             nn.PReLU()
#         )
#         self.layer = nn.Sequential(nn.Conv2d(64, 4, 9, stride=1, padding=4),
#                       nn.Tanh())# 范围为0到1)
#
#         self.gray = Gray(in_channel=4)
#
#         self.blur = kernel_spatial(n_layers, kw)
#
#         self.mse_loss = nn.MSELoss()
#         self.cfg = cfg
#     def forward(self,ms_lr_img ,ms_img, pan_img):
#         loss_cfg = self.cfg.get('loss_cfg', {})
#         x = torch.cat([ms_img, pan_img], dim=1)
#         ms_img_gray = self.gray(ms_img)
#         pan_img_blur = self.blur(torch.cat([pan_img for _ in range(4)], dim=1))
#         x0 = self.conv1(x)
#         x = self.residual_block(x0)
#         x = self.conv2(x)
#         x = self.conv3(x + x0)
#         output = self.layer(x)
#         output_blur = self.blur(output)
#         output_gray = self.gray(output)
#
#         fake_pan = torch.mean(output_gray, dim=1, keepdim=True)
#         fake_lr_up = output_blur
#
#         # reconstruction loss of generator
#         spatial_loss_rec = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
#         spectral_loss_rec = self.mse_loss(ms_lr_img, F.interpolate(output, scale_factor=0.25))
#
#         ms_img_gray = torch.mean(ms_img_gray, dim=1, keepdim=True)
#         pan_img_blur = torch.mean(pan_img_blur, dim=1, keepdim=True)
#         spatial_loss_RB = self.mse_loss(high_pass(ms_img_gray), high_pass(pan_img_blur))
#         spectral_loss_RB = self.mse_loss(fake_lr_up, ms_img)
#
#         G_loss =  loss_cfg['spatial_loss_rec'].w * spatial_loss_rec + loss_cfg['spectral_loss_rec'].w * spectral_loss_rec + \
#                   loss_cfg['spatial_loss_RB'].w*spatial_loss_RB  + loss_cfg['spectral_loss_RB'].w*spectral_loss_RB
#
#         return output,fake_pan,fake_lr_up,G_loss

class Discriminator_SRGAN(nn.Module):
    def __init__(self,in_channel):
        super(Discriminator_SRGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = discriminator(4).to(device)
# summary(model, (1,4, 128,128))

