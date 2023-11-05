# training settings
# model
import os
import torch
import datetime
#from models.model_TFnet import TFNet
# from models.model_fusionnet import FusionNet
# from models.model_LAGConv import LACNET
# from models.Pan_former import CrossSwinTransformer
from models.my_transformer import my_model_3_31_2
# from models.my_model_final import  my_model_4_6_3

# from models.model_MSDCNN import MSDCNN_model
# # from models.NLRNET import NLRNet
# from models.model_pannet import PanNet_model
# from models.GSA import GSA
#训练设置
test=False
tradition=False
cuda = True
MODEL=my_model_3_31_2()
model = 'my_model_3_31_2'
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
    test_STAMP='23-03-29-10'
    num_epochs = 120
elif model=='Panformer':
    ms_size=16
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-29-15'
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
    test_STAMP='23-03-29-10'
    num_epochs = 130
elif model=='LACNET':
    ms_size=64
    lr = 1e-4
    step =500
    decay_rate=0.1
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L2'
    #测试设置
    test_STAMP='23-03-29-11'
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
    test_STAMP='23-03-22-16'
    num_epochs = 150
elif model=='MSDCNN_model':
    ms_size=64
    lr = 1e-4
    step =60
    decay_rate=0.5
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L2'
    #测试设置
    test_STAMP='23-03-29-18'
    num_epochs = 130
elif model=='GSA':
    ms_size=16
    cuda=False
    num_epochs=0
    test_STAMP='23-03-28'
    loss_type='L2'
elif model=='my_model_3_13_ablation':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-30-09'
    num_epochs = 130
elif model=='my_model_3_13_easy':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-30-11'
    num_epochs = 130
elif model=='my_model_3_13_easy_ablation':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-30-11'
    num_epochs = 130
elif model=='my_model_3_30_4':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-03-30-19'
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
    test_STAMP='23-04-04-21'
    num_epochs = 200
elif model=='my_model_4_6_3':
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-04-06-13'
    num_epochs = 130


else:
    ms_size=64
    lr = 1e-4
    step =5
    decay_rate=0.99
    optimizer=torch.optim.Adam
    #损失函数
    loss_type='L1'
    #测试设置
    test_STAMP='23-04-02-10'
    num_epochs = 130

#数据集
dataset = 'GF_small' 
source='/media/dy113/disk1/Project_lzt/dataset/3 Gaofen-1'
#
#测试设置
TIMESTAMP=datetime.datetime.now().strftime('%y-%m-%d-%H')

test_batch_size = 5
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
train_type = 'train_low_res'# full是无监督 low是有监督
data_type = "tanh"

scale_factor = 4

device = torch.device("cuda:0" )
device_ids = [0]
parallel = True
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
            test_pair =41,
            train_pair = 0),
        test_data_2=dict(
            image_dirs=os.path.join(train_dir, dataset, test_type_2),
            source_path=source,
            stride=0,
            ms_size=0,
            pan_size=0,
            test_pair=41,
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
            stride=8,
            ms_size=16,
            pan_size=64,
            test_pair =5,
            train_pair = 16000),
        valid_data_2=dict(
            image_dirs=os.path.join(train_dir, dataset,valid_dir, test_type_2),
            source_path=source,
            stride=8,
            ms_size=16,
            pan_size=64,
            test_pair=5,
            train_pair=18000),
        train_data=dict(
            image_dirs=os.path.join(train_dir, dataset, train_type),
            source_path=source,
            stride=8,
            ms_size=16,
            pan_size=64,
            test_pair=41,
            train_pair=18000)
    )

