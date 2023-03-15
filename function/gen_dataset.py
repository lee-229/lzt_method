#处理PSData3数据集 进行降采样并切割图像
import numpy as np
import scipy.ndimage as snd
import scipy.misc as misc
from data_utils import load_image
import os
from data_utils import read_img2
# from resize_image import *
import scipy.io as scio
from utilis import auto_create_path
import cv2
#读取的文件类型为tiff文件

def get_image_id(filename):
    return filename.split('.')[0] #以-分割 并选定第一项

path = '/root/autodl-tmp/new_dataset/3 Gaofen-1'
type='Gaofen-1'
#原始文件的位置cd
source_ms_path = path+'/MS_256'
source_pan_path = path+'/PAN_1024'
#新文件的位置
source_RRms_path = path+'/MS_64'
source_RRpan_path = path+'/PAN_256'
#
def downsample(img):
    return cv2.resize( img, (64,64))
def downsample_pan(pan):
    return cv2.resize( pan, (256,256))
# def get_image_id(filename):
#     return filename.split('-')[0] #以-分割 并选定第一项
auto_create_path(source_RRms_path)
auto_create_path(source_RRpan_path)
#对原始图像进行降采样

import numpy as np
import torch
import torch.nn as nn
import math
import scipy.ndimage.filters as ft


def fspecial_gauss(size, sigma):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    # h = np.round(h, 4)
    return h


def fir_filter_wind(Hd, w):
    """
    compute fir filter with window method
    Hd:     desired freqeuncy response (2D)
    w:      window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)
    return h


def NyquistFilterGenerator(Gnyq, ratio, N):
    assert isinstance(Gnyq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'
    if isinstance(Gnyq, list):
        Gnyq = np.asarray(Gnyq)
    nbands = Gnyq.shape[0]

    kernel = np.zeros((N, N, nbands))  # generic kerenel (for normalization purpose)
    fcut = 1 / np.double(ratio)
    for j in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(Gnyq[j])))
        H = fspecial_gauss((N, N), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(N, 0.5)
        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))
    # kernel = np.round(kernel, 4)
    return kernel


def MTF(ratio, sensor, N=41):
    if (sensor == 'QB'):
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif (sensor == 'GF-1'):
        GNyq = np.asarray([0.208, 0.167, 0.174, 0.188])  # Bands Order: B,G,R,NIR
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif (sensor == 'GeoEye1') or (sensor == 'WV4'):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
    elif (sensor == 'WV2'):
        GNyq = 0.35 * np.ones((1, 7));
        GNyq = np.append(GNyq, 0.27)
    elif (sensor == 'WV3'):
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]

    h = NyquistFilterGenerator(GNyq, ratio, N)
    return h


def MTF_PAN(ratio, sensor, N=41):
    if (sensor == 'QB'):
        GNyq = np.array([0.15])
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')):
        GNyq = np.array([0.17])
    elif (sensor == 'GeoEye1') or (sensor == 'WV4'):
        GNyq = np.array([0.16])
    elif (sensor == 'WV2'):
        GNyq = np.array([0.11])
    elif (sensor == 'WV3'):
        GNyq = np.array([0.14])
    elif (sensor == 'GF-1'):
        GNyq = np.array([0.18])
    else:
        GNyq = np.array([0.15])
    return NyquistFilterGenerator(GNyq, ratio, N)


def interp23tap(img, ratio):
    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2, :] = img
        else:
            I1LRU[::2, ::2, :] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(I1LRU[:, :, i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ft.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img


def interp23tap_GPU(img, ratio):
    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
    BaseCoeff = np.expand_dims(BaseCoeff, axis=(0, 1))
    BaseCoeff = np.concatenate([BaseCoeff] * b, axis=0)

    BaseCoeff = torch.from_numpy(BaseCoeff)
    img = img.astype(np.float32)
    img = np.moveaxis(img, -1, 0)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros((b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c))

        if z == 0:
            I1LRU[:, 1::2, 1::2] = img
        else:
            I1LRU[:, ::2, ::2] = img

        I1LRU = np.expand_dims(I1LRU, axis=0)
        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=BaseCoeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = BaseCoeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(torch.from_numpy(I1LRU), 2, 3))
        img = conv(torch.transpose(t, 2, 3)).numpy()
        img = np.squeeze(img)

    img = np.moveaxis(img, 0, -1)

    return img


def wald_protocol(ms, pan, ratio, sensor, channels=4):
    mtf_kernel = MTF(ratio, sensor)

    MTF_kern = np.moveaxis(mtf_kernel, -1, 0)
    MTF_kern = np.expand_dims(MTF_kern, axis=1)
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)

    # DepthWise-Conv2d definition
    depthconv = nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=MTF_kern.shape,
                          groups=channels,
                          padding=20,
                          padding_mode='replicate',
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False

    ms_down = depthconv(ms)
    ms_wald_ = nn.functional.interpolate(ms_down, scale_factor=0.25, mode='bicubic')
    ms_lr = torch.zeros(ms.shape)
    for i in range(ms_wald_.shape[0]):
        temp = np.copy(np.asarray(torch.squeeze(torch.squeeze(ms_wald_[i, :, :, :]).permute((1, 2, 0))).detach().cpu()))
        ms_lr[i, :, :, :] = torch.from_numpy(interp23tap_GPU(temp, ratio)).permute((2, 0, 1))
    pan_lr = nn.functional.interpolate(pan, scale_factor=0.25, mode='bicubic')

    return ms_wald_, pan_lr

def wald_protocol_v2(ms, pan, ratio, sensor, channels=4):
    def genMTF_MS():
        mtf_kernel = MTF(ratio, sensor)

        MTF_kern = np.moveaxis(mtf_kernel, -1, 0)
        MTF_kern = np.expand_dims(MTF_kern, axis=1)
        MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)

        # DepthWise-Conv2d definition
        depthconv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=MTF_kern.shape,
                              groups=channels,
                              padding=20,
                              padding_mode='replicate',
                              bias=False)

        depthconv.weight.data = MTF_kern
        depthconv.weight.requires_grad = False

        ms_down = depthconv(ms)
        ms_wald_ = nn.functional.interpolate(ms_down, scale_factor=0.25, mode='bicubic')
        ms_lr = torch.zeros(ms.shape)
        for i in range(ms_wald_.shape[0]):
            temp = np.copy(
                np.asarray(torch.squeeze(torch.squeeze(ms_wald_[i, :, :, :]).permute((1, 2, 0))).detach().cpu()))
            ms_lr[i, :, :, :] = torch.from_numpy(interp23tap_GPU(temp, ratio)).permute((2, 0, 1))
        return ms_lr

    def genMTF_PAN():
        channels = 1
        mtf_kernel = MTF_PAN(ratio, sensor)

        MTF_kern = np.moveaxis(mtf_kernel, -1, 0)
        MTF_kern = np.expand_dims(MTF_kern, axis=1)
        MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)

        # DepthWise-Conv2d definition
        depthconv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=MTF_kern.shape,
                              groups=channels,
                              padding=20,
                              padding_mode='replicate',
                              bias=False)

        depthconv.weight.data = MTF_kern
        depthconv.weight.requires_grad = False

        pan_down = depthconv(pan)
        pan_lr = nn.functional.interpolate(pan_down, scale_factor=0.25, mode='bicubic')

        return pan_lr

    return genMTF_PAN()  # ms_lr, pan_lr
all_pan = []
all_ms = []
all_gt = []
for i in range(len(os.listdir(source_ms_path))):
    name = str(i+1)+'.mat' #1-500
    # MS= load_image(os.path.join(source_ms_path,i+'-MUL.tif')) #[C, H, W]
    # PAN = load_image(os.path.join(source_pan_path,i+'-PAN.tif')) #[4H, 4W]
    #为了解决高分一号数据集不一致的问题
    # print(i)
    # if i>105 and type=='Gaofen-1':
    #     MS = read_img2(os.path.join(source_ms_path,name),'I_MS')#[C,H,W]  
    #     PAN = read_img2(os.path.join(source_pan_path,name),'I_PAN')#[C,H,W]
    #     scio.savemat(os.path.join(source_pan_path,name), {'imgPAN': PAN})#H/4*W/4*C
    #     scio.savemat(os.path.join(source_ms_path,name), {'imgMS': MS})

    MS = read_img2(os.path.join(source_ms_path,name),'imgMS').transpose(2,0,1)#[C,H,W]
    PAN = read_img2(os.path.join(source_pan_path,name),'imgPAN')#[C,H,W]
    all_ms.append(MS)
    all_pan.append(PAN)
pan = torch.FloatTensor(all_pan).unsqueeze(1)
ms = torch.FloatTensor(all_ms)
print(pan.shape)
print(ms.shape)

I_MS_LR, I_PAN_LR = wald_protocol(ms, pan, 4, 'GF-1', channels=4)#[C,H,W]
for name in os.listdir(source_ms_path):
    #保存为.mat格式
    dataNew_ms = os.path.join(source_RRms_path, name)
    dataNew_pan = os.path.join(source_RRpan_path, name)
    i = get_image_id(name) #返回数值
    # 针对输入为h*w*c transpose
    scio.savemat(dataNew_ms, {'LRMS': I_MS_LR[int(i)-1,:,:,:].permute(1,2,0).numpy()})#H/4*W/4*C
    scio.savemat(dataNew_pan, {'LRPAN': I_PAN_LR[int(i)-1,:,:,:].squeeze(0).numpy()})#H*W