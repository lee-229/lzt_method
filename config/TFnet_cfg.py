# training settings
# model
import os
import torch
import datetime
test=False
model = 'my_model_3_9_2'
loss_type='L1'
lr = 1e-4
step = 5
decay_rate=0.99
optimizer=torch.optim.Adam
dataset = 'WV4_small' #makedata中的data source也要改
source='/root/autodl-tmp/new_dataset/4 WorldView-4/'
TIMESTAMP=datetime.datetime.now().strftime('%y-%m-%d-%H')
test_STAMP='23-03-09-21'
ms_size=64
# dataset
test_type = 'test_full_res'
test_type_2 = 'test_low_res'

valid_type = 'test_full_res'
valid_type_2 = 'test_low_res'
bit_depth = 11
# test savedir
savedir = './output/'
# loss
pixel_loss_type = 'L1'
# train
train_dir = './train/'
train_type = 'train_low_res'# full是无监督 low是有监督
data_type = "tanh"

scale_factor = 4
batch_size = 32
test_batch_size = 5
num_epochs = 200
start_epoch =21
resumeG = '/root/Pansharpening/Pansharpening/model_para/WV4_small/my_model_3_9_2/G/23-03-09-23/epoch20.pkl'
cuda = True
device = torch.device("cuda:0" )
device_ids = [0]
parallel = True
make_data = False
# resume
threads = 4

# test

stage1 = '/media/dy113/disk1/Project_lzt/code/LDP_GAN/model_para/WV4/Unet_cutblock_change /G/epoch100.pkl'
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
            stride=16,
            ms_size=32,
            pan_size=128,
            test_pair =50,
            train_pair = 10000),
        test_data_2=dict(
            image_dirs=os.path.join(train_dir, dataset, test_type_2),
            source_path=source,
            stride=16,
            ms_size=32,
            pan_size=128,
            test_pair=50,
            train_pair=10000),
        train_data=dict(
            image_dirs=os.path.join(train_dir, dataset, train_type),
            source_path=source,
            stride=16,
            ms_size=32,
            pan_size=128,
            test_pair=50,
            train_pair=10000)
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
            train_pair = 22000),
        valid_data_2=dict(
            image_dirs=os.path.join(train_dir, dataset,valid_dir, test_type_2),
            source_path=source,
            stride=8,
            ms_size=16,
            pan_size=64,
            test_pair=5,
            train_pair=22000),
        train_data=dict(
            image_dirs=os.path.join(train_dir, dataset, train_type),
            source_path=source,
            stride=8,
            ms_size=16,
            pan_size=64,
            test_pair=50,
            train_pair=22000)
    )

