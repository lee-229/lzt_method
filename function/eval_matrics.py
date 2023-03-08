import numpy as np
from scipy.ndimage import sobel
from numpy.linalg import norm
# from sewar.full_ref import sam, scc, ergas
from skimage import filters
import sewar as sewar_api
import torch
import cv2

def sam(ms, ps):
    assert ms.ndim == 3 and ms.shape == ps.shape

    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)

    dot_sum = np.sum(ms*ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)

    is_nan = np.nonzero(np.isnan(res))

    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0

    sam = np.mean(res)

    return sam * 180 / np.pi


def sCC(ms, ps):
    # ps_sobel = sobel(ps, mode='constant')
    # ms_sobel = sobel(ms, mode='constant')
    ps_sobel = np.zeros(ps.shape)
    ms_sobel = np.zeros(ps.shape)
    for i in range(ms.shape[2]):
        ps_sobel[:, :, i] = filters.sobel(ps[:, :, i])
        ms_sobel[:, :, i] = filters.sobel(ms[:, :, i])

    scc = np.sum(ps_sobel * ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel))

    return scc
def scc(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        #print(img1_[..., i].reshape[1, -1].shape)
        #test = np.corrcoef(img1_[..., i].reshape[1, -1], img2_[..., i].rehshape(1, -1))
        #print(type(test))
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')

def ergas(ms, ps, ratio=8):
    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)
    err = ms - ps
    ergas_index = 0
    for i in range(err.shape[2]):
        ergas_index += np.mean(np.square(err[:, :, i]))/np.square(np.mean(ms[:, :, i]))

    ergas_index = (100/ratio) * np.sqrt(1/err.shape[2]) * ergas_index

    return ergas_index



def D_lambda_numpy(l_ms, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_lambda value
    """
    if sewar:
        return sewar_api.d_lambda(l_ms, ps)

    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += np.abs(QIndex_numpy(ps[:, :, i], ps[:, :, j]) - QIndex_numpy(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)


def D_s_numpy(l_ms, pan, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_s value
    """
    if sewar:
        return sewar_api.d_s(pan, l_ms, ps)

    L = ps.shape[2]
    l_pan = cv2.pyrDown(pan)
    l_pan = cv2.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(QIndex_numpy(ps[:, :, i], pan) - QIndex_numpy(l_ms[:, :, i], l_pan))
    return sum / L

def D_lambda_torch(l_ms, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_lambda value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))
    return sum / L / (L - 1)


def D_s_torch(l_ms, pan, l_pan, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        pan (torch.Tensor): PAN images, shape like [N, C, H, W]
        l_pan (torch.Tensor): LR PAN images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_s value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        sum += torch.abs(QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum / L