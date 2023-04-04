#产生由拟合数据生成的训练集和验证集 返回值是numpy形式
import os
from os import listdir
from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io as scio
from skimage.io import imsave
import gdal
import mmcv
import imageio
import tifffile
import cv2
import torch
def auto_create_path(FilePath):
    if os.path.exists(FilePath):   ##目录存在，返回为真
            print( 'dir exists' )
    else:
            print( 'dir not exists')
            os.makedirs(FilePath)
def _is_lr_image(filename):
    return filename.endswith("pan.tif") #查看是否以这个为结尾 是则返回真

def get_image_id(filename):
    return filename.split('_')[0] #以-分割 并选定第一项
def load_image(path):
    """ Load .TIF image to np.array

    Args:
        path (str): path of TIF image
    Returns:
        np.array: value matrix in [C, H, W]
    """
    img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)
    return img
def write_img(filename,  im_data):

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff") 
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)       #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def read_img2(path, name):
    img = scio.loadmat(path)[name]  # 读取matlab数据 生成字典格式 H*W*C
    return img
def crop_to_patch(img, stride, all_img, name,ms_size ,pan_size):
    """
    输入分别为 h*w*c h*w*1
    """
    h = img.shape[0]
    w = img.shape[1]

    ratio = pan_size // ms_size
    if name == 'ms':
        for i in range(0, h - ms_size+stride , stride):
            for j in range(0, w - ms_size+ stride, stride):
                img_patch = img[i:i + ms_size , j:j + ms_size , :]
                all_img.append(img_patch)

    else:
        for i in range(0, h - pan_size+stride * ratio, stride * ratio):
            for j in range(0, w - pan_size+stride * ratio, stride * ratio):
                img_patch = img[i:i + pan_size, j:j + pan_size].reshape(pan_size, pan_size, 1)
                all_img.append(img_patch)

    return all_img

def tiff_save(img, img_name, img_path):
    #将图片保存为 c*h*w 和 h*w
    #img = img * 127.5 + 127.5  # 恢复到0-256
    save_path_1 = os.path.join(img_path, img_name)
    if img.ndim==3:
        img=img.transpose(2,0,1)
    img = img.squeeze()
   # img = img.astype('uint8')  # 从float转换成整形
    #tifffile.imsave(save_path_1, img)#将HWC自动转换成CHW 四波段图像使用
    write_img(save_path_1, img) 

def upsample(original_msi,scale):
    new_lrhs = []
    c = original_msi.shape[0]
    h = original_msi.shape[1]
    w = original_msi.shape[2]
    for i in range(c):
        temp = cv2.resize(original_msi[i,:,:],(h*scale,w*scale))
        temp = np.expand_dims(temp, 0) #在最后加入一个维度 变为【1000，1000，1】
        new_lrhs.append(temp)
    new_lrhs = np.concatenate(new_lrhs, axis=0)#把列表变成numpy array

    return new_lrhs

#产生数据集图片 (source_path='/media/dy113/disk1/Project_lzt/dataset/4 WorldView-4/')
class make_data():
    def __init__(self, cfg):
        self.image_dirs = cfg.image_dirs
        self.source_path = cfg.source_path
        self.stride = cfg.stride
        self.ms_size = cfg.ms_size
        self.pan_size = cfg.ms_size
        self.train_pair = cfg.train_pair
        self.test_pair = cfg.test_pair


    # image_dirs 保存剪裁后的图片的文件夹
    #   source_path 源图像的文件夹
    #   stride 剪裁的步幅
    def generate_data(self):
        auto_create_path(self.image_dirs)
        all_pan = []
        all_ms = []
        all_gt = []

        source_ms_path =self.source_path  + '/MS_256/'
        source_pan_path =self.source_path  + '/PAN_1024/'

        if self.image_dirs.endswith('train_full_res'):
            # source_RRms_path = self.source_path + '/MS_64/'
            # source_RRpan_path = self.source_path + '/PAN_256/'
            for mat_name in os.listdir(source_ms_path):
                #前test_pair张图片没有剪
                if int(mat_name.split('.')[0])>self.test_pair and len(all_pan)<=self.train_pair:
                    train_lrms= read_img2(source_ms_path+mat_name, 'imgMS') #H*W*C
                    train_pan = read_img2(source_pan_path+mat_name, 'imgPAN')
                    crop_to_patch(train_pan, self.stride, all_pan, 'pan', self.ms_size,self.pan_size)  # pan样本存入all_pan
                    crop_to_patch(train_lrms, self.stride, all_ms, 'ms',self.ms_size,self.pan_size)
            for i in range(len(all_pan)):
                tiff_save(all_pan[i], str(i) + '_pan.tif', self.image_dirs)
            for i in range(len(all_ms)):
                tiff_save(all_ms[i], str(i) + '_lr.tif', self.image_dirs)
            print('train data generated')
            print('The number of ms patch is: ' + str(len(all_ms)))  # ms样本的个数
            print('The number of pan patch is: ' + str(len(all_pan)))  # pan样本的个数

        if self.image_dirs.endswith ('test_full_res'):
            len_ms=0
            for mat_name in os.listdir(source_ms_path):
                if int(mat_name.split('.')[0]) <=self.test_pair:#每张图能剪出64张测试图片 其实只剪了五张就够了
                    # test_lrms = read_img2(source_ms_path + mat_name, 'imgMS')  # H*W*C
                    # test_pan = read_img2(source_pan_path + mat_name, 'imgPAN')
                    # crop_to_patch(test_pan, self.stride, all_pan, 'pan', self.ms_size, self.pan_size)  # pan样本存入all_pan
                    # crop_to_patch(test_lrms, self.stride, all_ms, 'ms', self.ms_size, self.pan_size)

                    # all_ms.append(read_img2(source_ms_path + mat_name, 'imgMS'))
                    # all_pan.append(read_img2(source_pan_path + mat_name, 'imgPAN'))
                    len_ms =len_ms+1
                    tiff_save(read_img2(source_ms_path + mat_name, 'imgMS'), mat_name.split('.')[0] + '_lr.tif', self.image_dirs)
                    tiff_save(read_img2(source_pan_path + mat_name, 'imgPAN'), mat_name.split('.')[0] + '_pan.tif', self.image_dirs)

                    
            # for i in range(len(all_pan)):
            #     tiff_save(all_pan[i], str(i) + '_pan.tif', self.image_dirs)
            # for i in range(len(all_ms)):
            #     tiff_save(all_ms[i], str(i) + '_lr.tif', self.image_dirs)
            print('test_full_res generated')
            print('The number of ms patch is: ' + str(len_ms))  # ms样本的个数
            print('The number of pan patch is: ' + str(len_ms))  # pan样本的个数
        if self.image_dirs.endswith ('test_low_res'):
            len_ms=0
            source_RRms_path = self.source_path + '/MS_64/'
            source_RRpan_path = self.source_path + '/PAN_256/'
            for mat_name in os.listdir(source_ms_path):
                if int(mat_name.split('.')[0]) <=self.test_pair: #选择第1 11 21...491张图片进行测试 共50张
                    # test_pan = read_img2(source_RRpan_path + mat_name, 'LRPAN')
                    # test_lrms = read_img2(source_RRms_path + mat_name, 'LRMS')
                    # gt = read_img2(source_ms_path + mat_name, 'imgMS')
                    # crop_to_patch(test_pan, self.stride, all_pan, 'pan', self.ms_size, self.pan_size)  # pan样本存入all_pan
                    # crop_to_patch(test_lrms, self.stride, all_ms, 'ms', self.ms_size, self.pan_size)
                    # crop_to_patch(gt, self.stride * 4, all_gt, 'ms', self.ms_size * 4, self.pan_size * 4)

                    #all_pan.append(read_img2(source_RRpan_path + mat_name, 'LRPAN'))
                    tiff_save(read_img2(source_RRpan_path + mat_name, 'LRPAN'), mat_name.split('.')[0] + '_pan.tif', self.image_dirs)
                    #all_ms.append(read_img2(source_RRms_path + mat_name, 'LRMS'))
                    tiff_save(read_img2(source_RRms_path + mat_name, 'LRMS'), mat_name.split('.')[0] + '_lr.tif', self.image_dirs)
                    #all_gt.append(read_img2(source_ms_path + mat_name, 'imgMS'))
                    tiff_save(read_img2(source_ms_path + mat_name, 'imgMS'), mat_name.split('.')[0]+ '_mul.tif', self.image_dirs)
                    len_ms=len_ms+1
            # for i in range(len(all_pan)):
            #     tiff_save(all_pan[i], str(i) + '_pan.tif', self.image_dirs)
            # for i in range(len(all_ms)):
            #     tiff_save(all_ms[i], str(i) + '_lr.tif', self.image_dirs)
            # for i in range(len(all_gt)):
            #     tiff_save(all_gt[i], str(i) + '_mul.tif', self.image_dirs)
            print('test_low_res generated')
            print('The number of ms patch is: ' + str(len_ms))  # ms样本的个数
            print('The number of pan patch is: ' + str(len_ms))  # pan样本的个数
            print('The number of gt patch is: ' + str(len_ms))  # pan样本的个数

        if self.image_dirs.endswith ('train_low_res'):
            source_RRms_path = self.source_path + '/MS_64/'
            source_RRpan_path = self.source_path + '/PAN_256/'
            for mat_name in os.listdir(source_ms_path):
                if int(mat_name.split('.')[0])>self.test_pair and len(all_pan)<=self.train_pair:
                    train_pan=read_img2(source_RRpan_path + mat_name, 'LRPAN')
                    train_lrms=read_img2(source_RRms_path + mat_name, 'LRMS')
                    gt=read_img2(source_ms_path + mat_name, 'imgMS')
                    
                    crop_to_patch(train_pan, self.stride, all_pan, 'pan', self.ms_size, self.pan_size)  # pan样本存入all_pan
                    crop_to_patch(train_lrms, self.stride, all_ms, 'ms', self.ms_size, self.pan_size)
                    crop_to_patch(gt, self.stride * 4, all_gt, 'ms', self.ms_size * 4, self.pan_size * 4)
            for i in range(len(all_pan)):
                tiff_save(all_pan[i], str(i) + '_pan.tif', self.image_dirs)
            for i in range(len(all_ms)):
                tiff_save(all_ms[i], str(i) + '_lr.tif', self.image_dirs)
            for i in range(len(all_gt)):
                tiff_save(all_gt[i], str(i) + '_mul.tif', self.image_dirs)
            print('train_low_res generated')
            print('The number of ms patch is: ' + str(len(all_ms)))  # ms样本的个数
            print('The number of pan patch is: ' + str(len(all_pan)))  # pan样本的个数
            print('The number of gt patch is: ' + str(len(all_gt)))  # pan样本的个数

#产生训练集
class TrainDatasetFromFolder(Dataset):
    def __init__(self, cfg):
        super(TrainDatasetFromFolder, self).__init__()
        self.dataset_dir = cfg.dataset_dir
        self.image_ids = []
        self.image_prefix_names = []  # full-path filename prefix
        self.norm_input = 2 ** cfg.input_bit
        self.type = cfg.data_type
        self.train_type = cfg.train_type
        for x in listdir(self.dataset_dir):
            if _is_lr_image(x) :
                self.image_ids.append(get_image_id(x))
                self.image_prefix_names.append(os.path.join(cfg.dataset_dir, get_image_id(x)))


    def __getitem__(self, index):
        #输出 n*c*h*w
        prefix_name = self.image_prefix_names[index]
        id = self.image_ids [index]
        if self.type =='sigmoid':
            lr_image = load_image('{}_lr.tif'.format(prefix_name)) / self.norm_input #转化成numpy格式
            pan_image = load_image('{}_pan.tif'.format(prefix_name)) /self.norm_input #转化成numpy格式 要给PAN增加维度
            lr_up_image = upsample(lr_image, 4)#C*H*W
            if self.train_type == 'train_low_res':
                target_image = load_image('{}_mul.tif'.format(prefix_name)) / self.norm_input  # 转化成numpy格式
            else:
                target_image = np.zeros(lr_up_image.shape)

        elif self.type =='tanh':
            lr_image =2*( load_image('{}_lr.tif'.format(prefix_name)) / self.norm_input)-1 #32*32
            pan_image = 2*(load_image('{}_pan.tif'.format(prefix_name)) /self.norm_input)-1 #128*128
            lr_up_image = upsample(lr_image, 4)#128*128

            # lrms_image = upsample(2 * (load_image('{}_lrms.tif'.format(prefix_name)) / self.norm_input) - 1 ,4) # 32*32
            # lrpan_image = 2 * (load_image('{}_lrpan.tif'.format(prefix_name)) / self.norm_input) - 1  # 32*32
            # gt_image = 2 * (load_image('{}_gt.tif'.format(prefix_name)) / self.norm_input) - 1  # 32*32

            if self.train_type == 'train_low_res':
                target_image = 2*(load_image('{}_mul.tif'.format(prefix_name)) / self.norm_input)-1  # 转化成numpy格式
            else:
                target_image = np.zeros(lr_up_image.shape)
        return lr_image.astype(np.float32), np.expand_dims(pan_image, axis=0).astype(np.float32), lr_up_image.astype(np.float32),\
               target_image.astype(np.float32)
               # lrms_image.astype(np.float32),np.expand_dims(lrpan_image, axis=0).astype(np.float32),gt_image.astype(np.float32)

    def __len__(self):
        return len(self.image_prefix_names)


#产生验证集
class TestDatasetFromFolder(Dataset):
    def __init__(self,cfg):
        super(TestDatasetFromFolder, self).__init__()
        self.dataset_dir = cfg.dataset_dir
        self.image_ids = []
        self.image_prefix_names = []  # full-path filename prefix
        self.norm_input = 2 ** cfg.input_bit
        self.test_type = cfg.test_type
        self.data_type = cfg.data_type
        for x in listdir(self.dataset_dir):
            if x.endswith("lr.tif") :
                self.image_ids.append(get_image_id(x))
            #筛选图片进入训练集
                self.image_prefix_names.append(os.path.join(cfg.dataset_dir, get_image_id(x)))


    def __getitem__(self, index):
        prefix_name = self.image_prefix_names[index]
        id = self.image_ids[index]
        if self.data_type == 'sigmoid':
            lr_image = load_image('{}_lr.tif'.format(prefix_name)) / self.norm_input #转化成numpy格式
            pan_image = load_image('{}_pan.tif'.format(prefix_name))/self.norm_input #转化成numpy格式
            lr_up_image = upsample(lr_image, 4)
            if self.test_type == 'test_low_res':
                target_image = load_image('{}_mul.tif'.format(prefix_name)) / self.norm_input  # 转化成numpy格式
            else:
                target_image = np.zeros(lr_up_image.shape)

        elif self.data_type =='tanh':
            lr_image =2*( load_image('{}_lr.tif'.format(prefix_name)) / self.norm_input) -1#转化成numpy格式
            pan_image = 2*(load_image('{}_pan.tif'.format(prefix_name)) /self.norm_input) -1#转化成numpy格式 要给PAN增加维度
            lr_up_image = upsample(lr_image, 4)
            if self.test_type == 'test_low_res':
                target_image = 2 * (load_image('{}_mul.tif'.format(prefix_name)) / self.norm_input) - 1  # 转化成numpy格式
            else:
                target_image = np.zeros(lr_up_image.shape)

        return lr_image.astype(np.float32), np.expand_dims(pan_image, axis=0).astype(np.float32), lr_up_image.astype(
            np.float32), target_image.astype(np.float32),id

    def __len__(self):
        return len(self.image_prefix_names)


def tiff_save_img(img,img_path,bit_depth,data_type='sigmoid'):
    if img.ndim == 2:
        # img = img.squeeze()
        if data_type =='sigmoid':
            img = img * 255
        elif data_type =='tanh':
            img = img * 127.5+127.5
        img = img.astype('uint8')  # 从float转换成整形
       # tifffile.imsave(img_path, img)
        write_img(img_path, img) 
    else:
        if data_type == 'sigmoid':
            img = img * 2047  # 恢复到0-2047
        else:
            img = img * 1023.5+1023.5

        N,M,c = img.shape[0],img.shape[1],img.shape[2]
        NM = N*M
        IMG = np.zeros((N, M, c))
        img= img.astype(np.uint16)
        #为了防止色彩失真 加入直方图均衡化
        for i in range(c):
            b = (img[:,:,i].reshape(NM,1))
            hb,levelb=np.histogram(b, bins=b.max()-b.min(), range=None)
            chb = np.cumsum(hb)
            t1 =levelb[np.where(chb > 0.04 * NM)[0][1]]
            t2_0 = np.where(chb <0.99 * NM)
            t2 = levelb[t2_0[0][np.size(t2_0[0]) - 1]]
            b[b < t1] = t1
            b[b > t2] = t2
            b = (b-t1)/(t2-t1)
            IMG[:,:,i] = b.reshape(N,M) * 255

        IMG = IMG.astype('uint8')  # 从float转换成整形
        if c==8:
            cv2.imwrite(img_path, IMG[:, :, [1,2,4]]) #读入顺序为BGR 与波段顺序一致
        if c==4:
            cv2.imwrite(img_path, IMG[:, :, [0,1,2]]) 
def tiff_save_img_no_his(img,img_path,bit_depth,data_type='sigmoid'):
    if img.ndim == 2:
        # img = img.squeeze()
        if data_type =='sigmoid':
            img = img * 255
        elif data_type =='tanh':
            img = img * 127.5+127.5
        img = img.astype('uint8')  # 从float转换成整形
        tifffile.imsave(img_path, img)
    else:
        if data_type == 'sigmoid':
            img = img * 255  # 恢复到0-2047
        else:
            img = img * 127.5+127.5
        img= img.astype(np.uint8)
        #为了防止色彩失真 加入直方图均衡化

        cv2.imwrite(img_path, img[:, :, [0,1,2]])
def visualize_img(img, data_type='sigmoid'):
    #针对三维数据
    if img.ndim == 2:
        # img = img.squeeze()
        if data_type == 'sigmoid':
            img = img * 255
        elif data_type == 'tanh':
            img = img * 127.5 + 127.5
        img_output = img.astype('uint8')  # 从float转换成整形
    else:
        if data_type == 'sigmoid':
            img = img * 255  # 恢复到0-2047
        else:
            img = img * 127.5 + 127.5
        N, M, c = img.shape[0], img.shape[1], img.shape[2]
        NM = N * M
        IMG = np.zeros((N, M, c - 1))
        img = img.astype(np.uint16)
        #为了防止色彩失真 加入直方图均衡化
        for i in range(3):
            b = (img[:, :, i].reshape(NM, 1))
            hb, levelb = np.histogram(b, bins=b.max() - b.min(), range=(b.min(), b.max())) #bins表示有多少个间距 range表示范围
            chb = np.cumsum(hb)
            t1 = np.where(chb > 0.04 * NM)[0][0]
            # np.where返回的是元组 [0]取其全部值
            t2 = np.where(chb > 0.99 * NM)[0][0]
            b[b < t1] = t1
            b[b > t2] = t2
            b = (b - t1) / (t2 - t1)
            IMG[:, :, i] = b.reshape(N, M) * 255
        IMG = IMG.astype('uint8')  # 从float转换成整形
        img_output = IMG[:, :, [2,1,0]] #读入顺序为RGB 与波段顺序一致
    return img_output

def zoom_in(img):
    #输入为三通道图片
    img = img.copy() #不加copy的话retangel会报错
    img_width = img.shape[0]
    img_height = img.shape[1]
    # 需要放大的部分
    x_start = int(0.1*img_width)
    y_start = int(0.1*img_width)
    mask_width = int(0.2*img_width)
    scale = 3
    mask_start_x = img_width - mask_width * scale
    mask_start_y = 0

    part = img[x_start:x_start + mask_width, y_start:y_start + mask_width]
    # 双线性插值法
    mask = cv2.resize(part, (scale * mask_width, scale * mask_width), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    # 放大后局部图的位置img[210:410,670:870]
    img[mask_start_x:mask_start_x + scale * mask_width, mask_start_y:mask_start_y + scale * mask_width] = mask

    # 画框并连线
    cv2.rectangle(img, (y_start, x_start), (y_start + mask_width, x_start + mask_width), (0, 255, 0), 4)
    cv2.rectangle(img, (mask_start_y, mask_start_x), (mask_start_y + scale * mask_width, mask_start_x + scale * mask_width),
                 (0, 255, 0), 4)
    return img
# 测试 tiff_save_img
# MS = 2*(read_img2(os.path.join('/media/dy113/disk1/Project_lzt/dataset/new_dataset/2 QuickBird/MS_256/1.mat'),'imgMS')/2047)-1#[C,H,W]
# tiff_save_img(MS,'./test.tif',11,'tanh' )
