import torch
import torch.nn as nn
import numpy as np
import numpy
from torchinfo import summary
import torch.nn.functional as F
from data_utils import load_image
import matplotlib.pyplot as plt
import cv2
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


class Generator(nn.Module):
    def __init__(self, scale=2):
        """放大倍数是scale的平方倍"""
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual_block = nn.Sequential(
            Block(),
            Block(),
            Block(),
            Block(),
            Block(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.PixelShuffle(scale),
            nn.PReLU(),
            #
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.PixelShuffle(scale),
            nn.PReLU()
        )
        self.conv4 = nn.Conv2d(64, 4, 9, stride=1, padding=4)

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.residual_block(x0)
        x = self.conv2(x)
        x = self.conv3(x + x0)
        x = self.conv4(x)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
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
# model = Discriminator().to(device)
# # summary(model, (1,4, 128,128))
# # input = torch.rand((1,4, 128,128))
# # input = input.cuda()
# # output = model(input)
# for i in range(1):
#     print(i)

# batches_img=torch.rand(1,2,5,5)#模拟图片数据（bs,2,4,4），通道数C为2
# print("batches_img:\n",batches_img)
#
# nn_Unfold=nn.Unfold(kernel_size=(3,3),dilation=1,padding=1,stride=1) #n*18*25
# patche_img=nn_Unfold(batches_img)#n*(k*k*c)*(h*w)
# patche_img = patche_img.reshape(1,18,5,5)#n*(k*k*c)*h*w
# img = torch.split(patche_img,2,dim=1) # list长度为k*k 每个元素为n*c*H*W
# img = torch.stack(img,dim=4)#n*c*H*W*(k*k)
# print("patche_img.shape:",img.shape)
# print("patch_img:\n",patche_img)

def image_hist(image): #画三通道图像的直方图
   color = ("red","green","blue")#画笔颜色的值可以为大写或小写或只写首字母或大小写混合
   for i, color in enumerate(color):
       hist = cv2.calcHist([image], [i], None, [256], [0, 256])
       plt.plot(hist, color=color)
       plt.xlim([0, 256])
   plt.show()
lr_image = load_image('/media/dy113/disk1/Project_lzt/code/LDP_GAN/train/WV4/test_full_res/0_lr.tif') #转化成numpy格式 #BGR 读进来的就是
img = lr_image/2047*255
img = img.astype(np.uint16)
NM = 256*256
IMG = np.zeros((256, 256, 3))
img = img.transpose(1,2,0)
image_hist(img)
for i in range(3):
    b = (img[:, :, i].reshape(NM, 1))
    hb, levelb = np.histogram(b, bins=b.max() - b.min(), range=(b.min(), b.max()))  # bins表示有多少个间距 range表示范围
    chb = np.cumsum(hb)
    t1 = np.where(chb > 0.1 * NM)[0][0]
    #np.where返回的是元组 [0]取其全部值
    t2 = np.where(chb > 0.99 * NM)[0][0]

    # b[b < t1] = t1
    # b[b > t2] = t2
    b = (b - t1) / (t2 - t1)
    IMG[:, :, i] = b.reshape(256, 256)
IMG = (IMG*255).astype(np.uint16)
image_hist(IMG)

#img_output = lr_image[[2, 1, 0],:, :]
plt.imshow(img[:,:,[2, 1, 0]])  #转成RGB
plt.show()
plt.imshow(IMG[:,:,[2, 1, 0]])  #转成RGB
plt.show()


