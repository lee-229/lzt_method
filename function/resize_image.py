# -*- coding: utf-8 -*-
"""
读取.mat文件中的MS 和PAN 输入分别为 c*h*w h*w
并按照传感器的MTF值对其进行降采样 输出为 h*w*c h*w*1
这一步和crop函数的输入格式相关
"""
import numpy as np
import scipy.ndimage as snd
import scipy.misc as misc
from function.data_utils import load_image
import os
from data_utils import read_img2
import scipy.io as scio
from utilis import auto_create_path
def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2)
    return w

def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h


def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)


def downgrade_images(I_MS, I_PAN, ratio, sensor):
    """
    downgrade MS and PAN by a ratio factor with given sensor's gains
    input：C*H*W
    """
    I_MS = np.double(I_MS)
    I_PAN = np.double(I_PAN)
    ratio = np.double(ratio)
    flag_PAN_MTF = 0

    if sensor == 'QB':

        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22], dtype='float32')  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif sensor == 'IKONOS':

        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28], dtype='float32')  # Band Order: B,G,R,NIR
        GNyqPan = 0.17;
    elif sensor == 'GeoEye1'or'WV4':

        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23], dtype='float32')  # Band Order: B,G,R,NIR
        GNyqPan = 0.16
    elif sensor == 'WV2':

        GNyq = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.27]
        GNyqPan = 0.11
    elif sensor == 'WV3':

        GNyq = 0.29 * np.ones(8)
        GNyqPan = 0.15

    N = 41
    I_MS_LP = np.zeros(I_MS.shape)
    fcut = 1 / ratio

    for j in range(I_MS.shape[0]):
        # fir filter with window method
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[j])))
        H = gaussian2d(N, alpha) #41*41的高斯滤波器
        Hd = H / np.max(H)
        w = kaiser2d(N, 0.5)
        h = fir_filter_wind(Hd, w)
        I_MS_LP[j, :, :] = snd.filters.correlate(I_MS[j, :, :], np.real(h), mode='nearest')

    if flag_PAN_MTF == 1:
        # fir filter with window method
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyqPan)))
        H = gaussian2d(N, alpha)
        Hd = H / np.max(H)
        h = fir_filter_wind(Hd, w)
        I_PAN = snd.filters.correlate(I_PAN, np.real(h), mode='nearest')
        I_PAN_LR = I_PAN[int(ratio / 2):-1:int(ratio), int(ratio / 2):-1:int(ratio)]

    else:
        # bicubic resize
        I_PAN_pad = np.pad(I_PAN, int(2 * ratio), 'symmetric')
        I_PAN_LR = misc.imresize(I_PAN_pad, 1 / ratio, 'bicubic', mode='F')
        I_PAN_LR = I_PAN_LR[2:-2, 2:-2]

    I_MS_LR = I_MS_LP[:, int(ratio / 2):-1:int(ratio), int(ratio / 2):-1:int(ratio)]

    return I_MS_LR, np.expand_dims(I_PAN_LR, axis=2)
#把1024和256的图片按照MTF值降采样到256和64
path = '/media/dy113/disk1/Project_lzt/dataset/new_dataset/2 QuickBird'
source_ms_path = path+'/MS_256'
source_pan_path = path+'/PAN_1024'
source_RRms_path = path+'/MS_64'
source_RRpan_path = path+'/PAN_256'
auto_create_path(source_RRms_path)
auto_create_path(source_RRpan_path)
for name in os.listdir(source_RRms_path):
    MS= read_img2(os.path.join(source_ms_path,name), 'imgMS') #256*256*4
    PAN = read_img2(os.path.join(source_pan_path,name), 'imgPAN') #64*64*1
    #针对输入为h*w*c 和h*w*1 transpose,squeeze()
    MS = MS.transpose(2,0,1) #4*64*64
    I_MS_LR, I_PAN_LR = downgrade_images(MS, PAN.squeeze(), 4, 'QB') ##4*64*64 1*256*256
    dataNew_ms = os.path.join(source_RRms_path, name)
    dataNew_pan = os.path.join(source_RRpan_path, name)
    # 针对输入为h*w*c transpose
    scio.savemat(dataNew_ms, {'LRMS': I_MS_LR.transpose(1,2,0)})#64*64*4
    scio.savemat(dataNew_pan, {'LRPAN': I_PAN_LR})#256*256*1




