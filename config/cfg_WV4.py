# training settings
# model
import os
import torch
import datetime
#from models.model_Unet import Unet_transformer_3_12
#from models.model_TFnet import TFNet
# # #from models.LDP_net import LDP_Net
#from models.model_MSDCNN import MSDCNN_model
from models.LDP_net import LDP_Net
# #from models.model_fusionnet import FusionNet
# from models.model_LAGConv import LACNET
#from models.Pan_former import CrossSwinTransformer
#from models.my_transformer import my_model_3_31_2,my_model_3_30_6
#from models.my_transformer_new import my_model_4_6_3
# from models.NLRNET import NLRNet
# from models.model_pannet import PanNet_model


# from models.Wavelet import Wavelet
# from models.SFIM import SFIM
# from models.MTF_GLP import MTF_GLP
#from models.Brovey import Brovey
# from models.IHS import IHS
#from models.PCA import GFPCA
#训练设置
test=False
tradition=False
cuda = True
MODEL=LDP_Net()
model = 'LDP_Net'
#batch size
batch_size = 32
#学习率
if model=='TFNet':
    ms_size=64
    lr = 1e-4
    step =250
    decay_rate=0.5
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-06-14'
    num_epochs = 100
elif model=='Panformer':
    ms_size=16
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-06-19'
    num_epochs = 130
elif model=='FusionNet':
    ms_size=64
    lr = 3e-4
    step =100
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L2'
    #测试设置
    test_STAMP='23-03-17-16'
    num_epochs = 130
elif model=='LACNET':
    ms_size=64
    lr = 1e-4
    step =100
    decay_rate=0.5
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L2'
    #测试设置
    test_STAMP='23-03-17-13'
    num_epochs = 120
elif model=='PanNet':
    ms_size=16
    lr = 1e-4
    step =100
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L2'
    #测试设置
    test_STAMP='23-03-18-13'
    num_epochs = 110
elif model=='Wavelet' or model=='IHS' or model=='MTF_GLP' or model=='GFPCA':
    ms_size=16
    cuda=False
    num_epochs=0
    test_STAMP='23-03-28'
    loss_type='L2'
elif model=='MSDCNN_model':
    ms_size=64
    lr = 1e-4
    step =60
    decay_rate=0.5
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L2'
    #测试设置
    test_STAMP='23-03-29-20'
    num_epochs = 100
elif model=='my_model_3_30_6':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-30-16'
    num_epochs = 150        
elif model=='my_model_3_31_2_no_res':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='SSIM+SAM'
    #测试设置
    test_STAMP='23-04-21-08'
    num_epochs = 140    
elif model=='my_model_3_31_2_no_CA':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='SSIM+SAM'
    #测试设置
    test_STAMP='23-04-21-13'
    num_epochs = 130       
elif model=='my_model_3_31_2':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='SSIM+SAM'
    #测试设置
    test_STAMP='23-04-20-13'
    num_epochs = 140 
elif model=='LDP_Net':
    in_nc = 4
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='unsuper_loss_LDP'
    #测试设置
    test_STAMP='23-10-10-19'
    num_epochs = 30         

else:
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='SSIM+SAM'
    #测试设置
    test_STAMP='23-04-02-20'
    num_epochs = 160

#数据集
dataset = 'WV4_small' 
source='/media/dy113/disk1/Project_lzt/dataset/4 WorldView-4'
#/media/dy113/disk1/Project_lzt/dataset/2 QuickBird
#测试设置
TIMESTAMP=datetime.datetime.now().strftime('%y-%m-%d-%H')

test_batch_size = 1
#恢复训练设置
start_epoch =1
resumeG = ''

test_type = 'test_full_res'
test_type_2 = 'test_low_res'
valid_type = 'test_full_res'
valid_type_2 = 'test_low_res'
bit_depth = 11
# test savedir
savedir = './output/'
# train
train_dir = './train/'
train_type = 'train_full_res'# full是无监督 low是有监督
data_type = "sigmoid"

scale_factor = 4

device = torch.device("cuda:0" )
device_ids = [0]
parallel = False
make_data = False
# resume
threads = 4

# test
pretrained = './model_para/'+dataset+'/'+model+'/G/'+test_STAMP+'/epoch'+str(num_epochs)+'.pkl'
valid_dir = "./valid"
csv_FR_dir = './test_result/'+model+'_'+dataset+'_'+TIMESTAMP+'_FR.csv'
csv_RR_dir = './test_result/'+model+'_'+dataset+'_'+TIMESTAMP+'_RR.csv'
test_dir = r""
tensorboard_path = 'tensorboard/'+model+'_'+dataset+'_'+TIMESTAMP
log_dir = f'logs/{model}/{dataset}/{TIMESTAMP}'
log_file = f'{log_dir}/.log'
log_level = 'INFO'
if(test==True):
    train_set_cfg = dict(
        dataset_train=dict(
            dataset_dir=os.path.join(train_dir, dataset, train_type),
            input_bit=bit_depth,
            train_type=train_type,
            data_type=data_type,
            valid=False),
        dataset_test=dict(
            dataset_dir=os.path.join(train_dir, dataset, test_type),
            input_bit=bit_depth,
            test_type=test_type,
            data_type=data_type),
        #用于测试降尺度数据
        dataset_test_2 = dict(
            dataset_dir=os.path.join(train_dir, dataset, test_type_2),
            input_bit=bit_depth,
            test_type=test_type_2,
            data_type=data_type))
    make_data_cfg = dict(
        test_data=dict(
            image_dirs=os.path.join(train_dir, dataset, test_type),
            source_path=source,
            stride=0,
            ms_size=0,
            pan_size=0,
            test_pair =50,
            train_pair = 0),
        test_data_2=dict(
            image_dirs=os.path.join(train_dir, dataset, test_type_2),
            source_path=source,
            stride=0,
            ms_size=0,
            pan_size=0,
            test_pair=50,
            train_pair=0),
        
    )
else:
    train_set_cfg = dict(
        dataset_train=dict(
            dataset_dir=os.path.join(train_dir, dataset, train_type),
            input_bit=bit_depth,
            train_type=train_type,
            data_type=data_type,
            valid=False),
        dataset_test=dict(
            dataset_dir=os.path.join(train_dir, dataset,valid_dir, test_type),
            input_bit=bit_depth,
            test_type=test_type,
            data_type=data_type),
        #用于测试降尺度数据
        dataset_test_2 = dict(
            dataset_dir=os.path.join(train_dir, dataset,valid_dir, test_type_2),
            input_bit=bit_depth,
            test_type=test_type_2,
            data_type=data_type))
    make_data_cfg = dict(
        valid_data=dict(
            image_dirs=os.path.join(train_dir, dataset, valid_dir,test_type),
            source_path=source,
            stride=32,
            ms_size=32,
            pan_size=128,
            test_pair =5,
            train_pair = 22000),
        valid_data_2=dict(
            image_dirs=os.path.join(train_dir, dataset,valid_dir, test_type_2),
            source_path=source,
            stride=32,
            ms_size=32,
            pan_size=128,
            test_pair=5,
            train_pair=22000),
        train_data=dict(
            image_dirs=os.path.join(train_dir, dataset, train_type),
            source_path=source,
            stride=32,
            ms_size=32,
            pan_size=128,
            test_pair=50,
            train_pair=11000)
    )

