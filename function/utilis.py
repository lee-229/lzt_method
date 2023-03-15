from functions import *
import os
import numpy as np
import torch
import math
from eval import D_s_numpy,D_lambda_numpy,SAM_numpy,ERGAS_numpy,Q4_numpy,SF_numpy,FCC_numpy
from eval_matrics import scc
# device = torch.device("cuda:1" )
from skimage import data,filters
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

def sobel_pass(img):
    device = img.device
    c=img.shape[1]
    sobel_kernel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
    sobel_kernel_x = sobel_kernel_x.reshape((1,1,3,3))
    sobel_kernel_x = sobel_kernel_x.repeat(c, c,1,1)
    sobel_kernel_x = sobel_kernel_x.to(device)
    edge_x = F.conv2d(img[:, :, :, :], sobel_kernel_x, padding='same', stride=1)
    # print(sobel_kernel_x.shape)
    # print(edge_x.shape)
    sobel_kernel_y = torch.tensor([[-1., -2., 1.], [0., 0., 0.], [1., 2., 1.]])
    sobel_kernel_y = sobel_kernel_y.reshape((1,1,3,3))
    sobel_kernel_y = sobel_kernel_y.repeat(c, c,1,1)
    sobel_kernel_y = sobel_kernel_y.to(device)
    # print(sobel_kernel_y.shape)
    edge_y = F.conv2d(img[:, :, :, :], sobel_kernel_y, padding='same', stride=1)
    # print(edge_y.shape)
    return torch.abs(edge_x)+torch.abs(edge_y)


def forward(self, X, Y):
    X_hx = self.conv_hx(X)
    X_hy = self.conv_hy(X)
    G_X = torch.abs(X_hx) + torch.abs(X_hy)
    # compute gradient of Y
    Y_hx = self.conv_hx(Y)
    self.conv_hx.train(False)
    Y_hy = self.conv_hy(Y)
    self.conv_hy.train(False)
    G_Y = torch.abs(Y_hx) + torch.abs(Y_hy)



def generate_Gauss(kernel_size,gaussian_variance):
    x = np.arange(0, kernel_size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = kernel_size // 2
    kernel=np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / gaussian_variance/ gaussian_variance)
    kernel=kernel/np.sum(kernel)
    return kernel
def gaussianBlur(img,kernel_size,gaussian_variance):
    #输入 n*4*h*w 输出n*4*h*w
    print(kernel_size)
    kernel_size = math.ceil(kernel_size.item())
    print(kernel_size)
    print(gaussian_variance)
    gaussian_variance = math.ceil(gaussian_variance.item())
    print(gaussian_variance)
    kernel=torch.zeros(1, 1, kernel_size, kernel_size)
    kernel = kernel.cuda()
    value = generate_Gauss(kernel_size,gaussian_variance)
    print(kernel)
    print(value)
    kernel[0, 0, :, :] = torch.tensor(value)
    print(kernel[0, 0, :, :])

    x1 = img[:, 0, :, :]
    x2 = img[:, 1, :, :]
    x3 = img[:, 2, :, :]
    x4 = img[:, 3, :, :]
    x1 = F.conv2d(x1.unsqueeze(1), kernel, padding="same", stride=1)
    x2 = F.conv2d(x2.unsqueeze(1), kernel, padding="same", stride=1)
    x3 = F.conv2d(x3.unsqueeze(1), kernel, padding="same", stride=1)
    x4 = F.conv2d(x4.unsqueeze(1), kernel, padding="same", stride=1)
    img_blur = torch.cat([x1, x2, x3, x4], dim=1)
    return img_blur


def torch2np(data):
    r""" transfer image from torch.Tensor to np.ndarray

    Args:
        data (torch.Tensor): image shape like [N, C, H, W]
    Returns:
        np.ndarray: image shape like [N, H, W, C] or [N, H, W]
    """
    if data.shape[1] is 1:
        return data.squeeze(1).cpu().detach().numpy()
    else:
        return data.cpu().detach().numpy().transpose(0, 2, 3, 1)
def auto_create_path(FilePath):
    if os.path.exists(FilePath):   ##目录存在，返回为真
            print( 'dir exists' )
    else:
            print( 'dir not exists')
            os.makedirs(FilePath)
#初始学习率为1e-4，每隔20代学习率衰减为原来的0.1倍
def adjust_learning_rate(epoch,lr,step,decay_rate):
    lr = lr * (decay_rate ** (epoch // step))
    if lr < 1e-6:
        lr = 1e-6
    return lr

def save_checkpoint(module, module_name,dataset,model_name,epoch,time):
    '''

    :param module: G  or D
    :param module_name: 'G'  or 'D'
    :param dataset: 数据集名称
    :param epoch: 代数
    :param model_name:模型名称
    :return:
    '''
    model_folder = os.path.join("./model_para/", dataset,model_name,module_name,time)
    auto_create_path(model_folder)
    model_parm_path = os.path.join(model_folder,"epoch{}.pkl".format(epoch))
    # model_state = {"epoch": epoch, "model": model}
    torch.save(module.state_dict(), model_parm_path)
    print("Checkpoint saved to {}".format(model_parm_path))
# 计算测试结果

def eval_compute(input_pan,input_lr,pansharpening,target,test_type,data_type,logger):
    #计算融合图像的数值指标并打印出来
    """
     test_type = 'test_low_res'时为降尺度评估 ='test_full_res'时为全尺度评估
     data_type为数据的范围 即0-1或者-1 - 1
     """
    # torch to np
    input_pan = torch2np(input_pan)  # shape of [N, H, W]
    input_lr = torch2np(input_lr)  # shape of [N, H, W, C]
    pansharpening = torch2np(pansharpening)
    target = torch2np(target)
    tmp_results = {}
    eval_results = {}
    if test_type == 'test_low_res':
        eval_metrics = ['SAM', 'ERGAS', 'Q4', 'SCC']
    else:
        eval_metrics = ['D_lambda', 'D_s', 'QNR','SF','FCC']

    for metric in eval_metrics:
        tmp_results.setdefault(metric, [])
    # batch_size的循环
    for i in range(pansharpening.shape[0]):
        # img_name = str(image_index[i]) + '.tif'
        # tiff_save_img(output_np[i], os.path.join(save_pre_dir, img_name), cfg.bit_depth,
        #               data_type='tanh')  # 先转换成numpy 再保存RGB
        # tiff_save_img(input_lr[i], os.path.join(save_ms_dir, img_name), cfg.bit_depth,
        #               data_type='tanh')  # 先转换成numpy 再保存RGB
        # tiff_save_img(input_pan[i], os.path.join(save_pan_dir, img_name), cfg.bit_depth,
        #               data_type='tanh')  # 先转换成numpy 再保存RGB
        if test_type == 'test_full_res':
            if data_type == 'tanh':
                tmp_results['D_lambda'].append(D_lambda_numpy(input_lr[i]/ 2 + 0.5, pansharpening[i]/ 2 + 0.5, sewar=False))
                tmp_results['D_s'].append(D_s_numpy(input_lr[i]/ 2 + 0.5, input_pan[i]/ 2 + 0.5, pansharpening[i]/ 2 + 0.5, sewar=False))
                tmp_results['QNR'].append((1 - tmp_results['D_lambda'][-1]) * (1 - tmp_results['D_s'][-1]))
                tmp_results['SF'].append(SF_numpy(pansharpening[i]/2+0.5))
                tmp_results['FCC'].append(FCC_numpy(input_pan[i]/ 2 + 0.5,pansharpening [i]/ 2 + 0.5))

            else:
                tmp_results['D_lambda'].append(
                    D_lambda_numpy(input_lr[i] , pansharpening[i] , sewar=False))
                tmp_results['D_s'].append(
                    D_s_numpy(input_lr[i] , input_pan[i] , pansharpening[i] , sewar=False))
                tmp_results['QNR'].append((1 - tmp_results['D_lambda'][-1]) * (1 - tmp_results['D_s'][-1]))
                tmp_results['SF'].append(SF_numpy(pansharpening[i] ))
                tmp_results['FCC'].append(FCC_numpy(input_pan[i], pansharpening[i]))
            # image = torch.transpose(image, 1, 3)
        if test_type == 'test_low_res':
            if data_type == 'tanh':
                tmp_results['SAM'].append(SAM_numpy(target[i]/ 2 + 0.5, pansharpening[i]/ 2 + 0.5, sewar=False))
                tmp_results['ERGAS'].append(ERGAS_numpy(target[i]/ 2 + 0.5, pansharpening[i]/ 2 + 0.5, sewar=False))
                tmp_results['Q4'].append(Q4_numpy(target[i]/ 2 + 0.5, pansharpening[i]/ 2 + 0.5))
                tmp_results['SCC'].append(scc(target[i]/ 2 + 0.5, pansharpening[i]/ 2 + 0.5))
            else:
                tmp_results['SAM'].append(SAM_numpy(target[i], pansharpening[i], sewar=False))
                tmp_results['ERGAS'].append(ERGAS_numpy(target[i], pansharpening[i], sewar=False))
                tmp_results['Q4'].append(Q4_numpy(target[i], pansharpening[i]))
                tmp_results['SCC'].append(scc(target[i], pansharpening[i]))

    #计算平均值 打印结果
    for metric in eval_metrics:
        eval_results.setdefault(f'{metric}_mean', [])
        eval_results.setdefault(f'{metric}_std', [])
        mean = np.mean(tmp_results[metric])
        std = np.std(tmp_results[metric])
        eval_results[f'{metric}_mean'].append(round(mean, 4))
        eval_results[f'{metric}_std'].append(round(std, 4))
        #logger.info(f'{metric} metric value: {mean:.4f} +- {std:.4f}')
    return tmp_results




