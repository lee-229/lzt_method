import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
from function.eval import D_lambda_torch,D_s_torch,SAM_torch
from function.utilis import high_pass,sobel_pass
from pytorch_msssim import  SSIM
from torch.nn.functional import cosine_similarity
from torchinfo import summary
from collections import OrderedDict
device = torch.device("cuda:1" )
class D_loss(nn.Module):
    # 无监督学习的D loss
    def __init__(self):
        super(D_loss, self).__init__()
        self.mse_loss = nn.L1Loss()

    def forward(self, spatial_pos, spatial_neg, spectral_pos,spectral_neg):
        fake_score = torch.zeros(spatial_pos.shape).to(device)
        valid_score = torch.ones(spatial_pos.shape).to(device)
        # loss of Discrinator
        # spatial loss
        spatial_pos_loss = self.mse_loss(spatial_pos, valid_score)
        spatial_neg_loss = self.mse_loss(spatial_neg, fake_score)
        spatial_D_loss = spatial_pos_loss + spatial_neg_loss

        # spectral loss
        spectral_pos_loss = self.mse_loss(spectral_pos, valid_score)
        spectral_neg_loss = self.mse_loss(spectral_neg, fake_score)
        spectral_D_loss = spectral_pos_loss + spectral_neg_loss
        return spatial_D_loss,spectral_D_loss
               #+ 2e-8 * tv_loss#去掉正则项
class D_loss_super(nn.Module):
    # 有监督学习的D loss
    def __init__(self):
        super(D_loss_super, self).__init__()
        self.mse_loss = nn.L1Loss()

    def forward(self, spatial_pos, spatial_neg, spectral_pos,spectral_neg):
        fake_score = torch.zeros(spatial_pos.shape).to(device)
        valid_score = torch.ones(spatial_pos.shape).to(device)
        # loss of Discrinator
        # spatial loss
        spatial_pos_loss = self.mse_loss(spatial_pos, valid_score)
        spatial_neg_loss = self.mse_loss(spatial_neg, fake_score)
        spatial_D_loss = spatial_pos_loss + spatial_neg_loss

        # spectral loss
        spectral_pos_loss = self.mse_loss(spectral_pos, valid_score)
        spectral_neg_loss = self.mse_loss(spectral_neg, fake_score)
        spectral_D_loss = spectral_pos_loss + spectral_neg_loss
        return spatial_D_loss,spectral_D_loss
               #+ 2e-8 * tv_loss#去掉正则项
class G_loss_adv(nn.Module):
    # pangan的G 生成对抗loss
    def __init__(self):
        super(G_loss_adv, self).__init__()
        self.mse_loss = nn.L1Loss()

    def forward(self,  spatial_neg,spectral_neg):

        valid_score1 = torch.ones(spectral_neg.shape).to(device)
        valid_score2 = torch.ones(spatial_neg.shape).to(device)


        spatial_G_loss = self.mse_loss(spatial_neg, valid_score2)
        spectral_G_loss = self.mse_loss(spectral_neg, valid_score1)
        G_loss_adv = 5 * spatial_G_loss + spectral_G_loss
        return G_loss_adv

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        vgg = vgg16(pretrained=True)
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1)  # 调整VGG为单通道输入


        loss_network = nn.Sequential(*list(vgg.features)[:11]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

    def forward(self,  out_images,target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # perception_loss = self.loss_network(out_images)
        return  perception_loss

class QNRLoss(nn.Module):
    def __init__(self):
        super(QNRLoss, self).__init__()


    def forward(self, pan, ms, out, pan_l=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        D_lambda = D_lambda_torch(l_ms=ms/ 2 + 0.5, ps=out/ 2 + 0.5)
        D_s = D_s_torch(l_ms=ms/ 2 + 0.5, pan=pan/ 2 + 0.5, l_pan=pan_l/ 2 + 0.5 if pan_l is not None else down_sample(pan/ 2 + 0.5), ps=out/ 2 + 0.5)
        QNR = (1 - D_lambda) * (1 - D_s)
        return 1 - QNR

def down_sample(imgs, r=4, mode='bicubic'):
    r""" down-sample the images

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        r (int): scale ratio, Default: 4
        mode (str): interpolate mode, Default: 'bicubic'
    Returns:
        torch.Tensor: images after down-sampling, shape of [N, C, H//r, W//r]
    """
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h // r, w // r], mode=mode, align_corners=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = perception_loss().to(device)
# summary(model, ((1,1, 32,32),(1,1, 32,32)))
class unsuper_loss(nn.Module):
    def __init__(self):
        super(unsuper_loss, self).__init__()
        self.mse_loss = nn.L1Loss()
        self.QNR = QNRLoss()


    def forward(self, lr_ms_img,pan_img,pansharpening,super_output):

        fake_pan =torch.mean(pansharpening,dim=1,keepdim=True)
        fake_grad = sobel_pass(fake_pan)
        real_grad = sobel_pass(pan_img)
        spatial_loss = self.mse_loss(fake_grad, real_grad)
        loss_QNR= self.QNR(pan_img, lr_ms_img, pansharpening)
        #spectral_loss_lr =self.mse_loss(pansharpening,  lr_ms_img)
        #SAM_loss = SAM_torch(super_output, pansharpening)
        spectral_loss = self.mse_loss(pansharpening, super_output)
        loss =spatial_loss +10*spectral_loss +5*loss_QNR

        return loss
class unsuper_loss_QNR(nn.Module):
    def __init__(self):
        super(unsuper_loss_QNR, self).__init__()
        self.mse_loss = nn.L1Loss()
        self.QNR = QNRLoss()


    def forward(self, lr_ms_img,pan_img,pansharpening,super_output):

        loss_new = self.QNR(pan_img,lr_ms_img,pansharpening)
        loss_reg = self.mse_loss(pansharpening, super_output)
        loss =10*loss_new + loss_reg

        return loss
def criterion(self, output, _label):
    spatital_loss = self.l1(output, _label) * 85
    spectral_loss = torch.mean(1 - cosine_similarity(output, _label, dim=1)) * 15

    # band shuffle
    sq = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 0]).type(torch.LongTensor)
    # shuffle real_img
    base = _label[:, sq, :, :]
    new_label = _label - base
    # shuffle fake_img
    base = output[:, sq, :, :]
    new_fake = output - base
    spectral_loss2 = self.l1(new_label, new_fake) * 15

    return spatital_loss + spectral_loss + spectral_loss2
class super_loss(nn.Module):
    def __init__(self,loss_type=None):
        super(super_loss, self).__init__()
        self.loss=loss_type
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()


    def forward(self, target,pansharpening):
        if (self.loss == 'L1'):
            return self.L1(pansharpening,target)
        elif(self.loss == 'L2'):
            return self.L2(target,pansharpening)
        #SAM_loss =SAM_torch(target,pansharpening)
        #return 3*Pixel_loss + SAM_loss

class T_net_loss(nn.Module):
    #输入为n*1*128*128 用于计算拟合的PAN和真实PAN的梯度差距
    def __init__(self):
        super(T_net_loss, self).__init__()
        self.loss =nn.L1Loss()
        self.ssim_module =SSIM(data_range=1, size_average=True, channel=1)
        # self.qnr = QNRLoss()

    def forward(self, pan_img,out):
        # high_pass_loss=self.mse_loss(high_pass(pan_img),high_pass(out))
        #structrue_loss = 1-self.ssim_module(pan_img/2+0.5, out/2+0.5)
        structrue_loss=self.loss(pan_img, out)
        return 100*structrue_loss

class G_loss_unet_adv(nn.Module):
    # unet的G 生成对抗loss
    # D_value为鉴别器给生成图像的值
    def __init__(self):
        super(G_loss_unet_adv, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self,  D_value):

        label = torch.ones(D_value.shape).to(device)
        G_loss_adv = self.mse_loss(D_value, label) #采用LSGAN的结构
        #G_loss_adv = -torch.mean(D_value)
        return G_loss_adv
class D_loss_unet(nn.Module):
    # 无监督学习的D loss
    def __init__(self):
        super(D_loss_unet, self).__init__()
        self.mse_loss = nn.MSELoss() #LSGAN

    def forward(self, pos_value,neg_value):
        fake_score = torch.zeros(neg_value.shape).to(device)
        valid_score = torch.ones(pos_value.shape).to(device)
        # loss of Discrinator
        pos_loss = self.mse_loss(pos_value, valid_score)
        neg_loss = self.mse_loss(neg_value, fake_score)
        D_loss = pos_loss + neg_loss
        #D_loss = torch.mean(neg_value) - torch.mean(pos_value)
        return D_loss