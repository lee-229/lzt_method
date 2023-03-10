import torch
import torch.nn as nn
import math
from torchinfo import summary
from function.utilis import high_pass,sobel_pass
from function.functions import *
def local_conv(img, kernel_2d,  kernel_size):
    """
    进行局部卷积 输入
        """
    n = img.shape[0]
    c = img.shape[1]
    h = img.shape[2]
    w = img.shape[3]
    img = roll(img,kernel_size)

    kernel_2d = kernel_2d.transpose(1,2).transpose(2,3) #n*H*W*(k*k)
    kernel_2d = kernel_2d.unsqueeze(1)#n*1*H*W*(k*k)
    kernel_2d = kernel_2d.repeat(1,c,1,1,1)#n*c*H*W*(k*k)
    result = torch.mul(img,kernel_2d)#n*c*H*W*(k*k)
    result = result.sum(dim=4)#n*c*H*W
    return result
def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor
def roll(img,kernel_size):
    """
        Implementation of rolling a img to different offsets depend on kernel_size
        Parameters:
        -----------
        img: input tensor, shape is [N, C, H, W]
        kernel_size: if kernel_size = 9 img will be shifted from (-4,4) to (4,4)
        Returns:
        --------
        tensor: tensor after shifted, shape is [N, C, H, W,k*k],
        """
    n = img.shape[0]
    c = img.shape[1]
    h = img.shape[2]
    w = img.shape[3]
    nn_Unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), dilation=1, padding=(kernel_size - 1) // 2, stride=1)
    imgs = torch.split(img, 1, dim=1)  # list长度为c 每个元素为n*1*H*W
    IMG = []
    for i in range(c):
        IMG.append(nn_Unfold(imgs[i])) # n*(k*k)*(h*w)
        IMG[i] = IMG[i].reshape(n, kernel_size * kernel_size , h, w)  # n*(k*k)*h*w
        IMG[i] = torch.split(IMG[i], 1, dim=1)  # list长度为k*k 每个元素为n*1*H*W
        IMG[i] = torch.stack(IMG[i], dim=4)  # n*1*H*W*(k*k)
    roll_img = torch.cat(IMG, dim=1)# n*c*H*W*(k*k)
    return roll_img
def sis_loss(img_align,img_MS,kernel_size):
    ## input n*c*H*W
    img_align_list = roll(img_align, kernel_size)#n*c*H*W*(k*k)
    img_MS_list = img_MS.unsqueeze(-1).repeat(1,1,1,1,kernel_size*kernel_size)#n*c*H*W*(k*k)

    p_dist = torch.abs(img_align_list-img_MS_list)#n*c*H*W*(k*k) 按元素求差
    p_dist_min =torch.min(p_dist,dim=-1).values ##n*c*H*W 最小值
    loss = torch.mean(torch.square(p_dist_min))
    return loss
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU()
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.lrelu(residual)
        # residual = self.conv2(residual)
        return x + residual
class FAM(nn.Module):
    def __init__(self,kernel_size):
        super(FAM, self).__init__()
        kernel_size = 3
        self.AFX = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.LeakyReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, kernel_size*kernel_size, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            ResidualBlock(kernel_size*kernel_size),
            # ResidualBlock(kernel_size*kernel_size),
            nn.Softmax()
        )
        self.MFX = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.reconstruct = nn.Sequential( nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
                                          nn.LeakyReLU()  )
        self.kernel_size = kernel_size

    def forward(self, ms_img, pan_img):
        lr_pan = pixel_shuffle_inv(pan_img, 4) #space to channel n*(4*4*1)*H*W
        #input_AFX = torch.cat([ms_img,lr_pan],dim=1)#n*20*H*W
        #offset_map = self.AFX(input_AFX)#n*81*H*W
        MS_F = self.MFX(ms_img)#n*16*H*W
        #aligned_MS_F = local_conv(MS_F,offset_map,kernel_size=self.kernel_size)
        return self.reconstruct(MS_F)#n*64*H*W
        # return aligned_MS_F  # n*64*H*W
class SIPSA(nn.Module):
    def __init__(self,kernel_size=7):
        super(SIPSA, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ResidualBlock(32)
        )
        self.add =  self.layer3_pan = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.kernel_size=kernel_size
        self.mse_loss = nn.MSELoss()
        self.align = FAM(kernel_size)

    def forward(self,ms_img,ms_lr_img,pan_img):
        align_ms = self.align(ms_lr_img,pan_img)
        align_ms_up = F.interpolate(align_ms, scale_factor=4)
        lr_pan = F.interpolate(pan_img, scale_factor=0.25)

        input = torch.cat([align_ms_up,pan_img],dim=1)
        out_128 = self.extract(input)
        out = self.add(out_128)+align_ms_up
        #
        fake_pan = torch.mean(out, dim=1, keepdim=True)
        # spatial_loss
        spatial_loss_hr = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
        spatial_loss_lr = self.mse_loss(high_pass(lr_pan), torch.mean(high_pass(align_ms), dim=1, keepdim=True))
        # sisloss
        #sisloss_lr = sis_loss(align_ms,ms_lr_img,self.kernel_size)
        #sisloss_hr = sis_loss(align_ms_up,ms_img,self.kernel_size)
        #spectral loss
        spectral_loss = self.mse_loss(align_ms,ms_lr_img)
        spatial_loss_rec = self.mse_loss(high_pass(pan_img), high_pass(fake_pan))
        spectral_loss_rec = self.mse_loss(F.interpolate(out, scale_factor=0.25), ms_lr_img)
        total_loss = 5 * spatial_loss_rec + spectral_loss_rec +spectral_loss
        #total_loss = spectral_loss
        # return align_ms,out,total_loss
        return align_ms,out,total_loss

# n=5
# img = torch.Tensor([[[[x * n + y + 1] for y in range(n)] for x in range(n)]for x in range(3)]).reshape(1,3,5,5)
# img2 = img+1
# loss = sis_loss(img,img2,3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = SIPSA().to(device)
# summary(model, ((1,4, 128,128),(1,4, 32,32),(1,1, 128,128)))

