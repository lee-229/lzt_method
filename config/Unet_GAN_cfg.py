# training settings
# model
import os
import torch

model = 'Unet_GAN'
# dataset
dataset = 'WV4' #makedata中的data source也要改
test_type = 'test_full_res'
test_type_2 = 'test_low_res'
bit_depth = 11
# test savedir
savedir = './output/'
# loss
pixel_loss_type = 'L1'
# train
train_dir = './train/'
train_type = 'train_low_res'# full是无监督 low是有监督
data_type = "tanh"
lr = 1e-4
scale_factor = 4
batch_size = 32
test_batch_size = 1
num_epochs = 100
cuda = True
device = torch.device("cuda:1" )
device_ids = [1,2]
parallel = True
make_data = False
# resume
resumeG =''
start_epoch = 1
threads = 4
step = 20
# test
test = True
pretrained = './model_para/WV4/Unet_GAN/G/epoch100.pkl'
train_dir = "./train/"
csv_FR_dir = model+'_FR.csv'
csv_RR_dir = model+'_RR.csv'
test_dir = r""
tensorboard_path = 'tensorboard/'+model
log_dir = f'logs/{model.lower()}'
log_file = f'{log_dir}/{model}.log'
log_level = 'INFO'
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
loss_cfg = dict(
    spatial_loss_high=dict(w=5),
    spectral_loss=dict(w=1),
    spatial_loss_low=dict(w=0.5),
    QNR_loss = dict(w=0.5)
)


make_data_cfg = dict(
    test_data=dict(
        image_dirs=os.path.join(train_dir, dataset, test_type),
        source_path="/media/dy113/disk1/Project_lzt/dataset/new_dataset/4 WorldView-4",
        stride=16,
        ms_size=32,
        pan_size=128,
        test_pair =20,
        train_pair = 10000),
    test_data_2=dict(
        image_dirs=os.path.join(train_dir, dataset, test_type_2),
        source_path="/media/dy113/disk1/Project_lzt/dataset/new_dataset/4 WorldView-4",
        stride=16,
        ms_size=32,
        pan_size=128,
        test_pair=20,
        train_pair=10000),
    train_data=dict(
        image_dirs=os.path.join(train_dir, dataset, train_type),
        source_path="/media/dy113/disk1/Project_lzt/dataset/new_dataset/4 WorldView-4",
        stride=16,
        ms_size=32,
        pan_size=128,
        test_pair=5,
        train_pair=10000)
)
