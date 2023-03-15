# training settings
# model
import os
import torch
import datetime
#from models.model_Unet import Unet_transformer_3_12
from models.model_TFnet import TFNet
# #from models.LDP_net import LDP_Net
# from models.model_MSDCNN import MSDCNN_model
# from models.model_fusionnet import FusionNet
# from models.model_LAGConv import LACNET
# from models.Pan_former import CrossSwinTransformer
from models.my_transformer import my_model_3_11_4,my_model_3_10,my_model_3_13,my_model_3_13_2
# from models.NLRNET import NLRNet
#训练设置
test=False
MODEL=TFNet()
model = 'TFNet'
#batch size
batch_size = 32
#学习率
lr = 1e-4
step =250
decay_rate=0.5
optimizer=torch.optim.Adam
#损失函数
loss_type='L1'
#数据集
dataset = 'GF_1_small' #makedata中的data source也要改
source='/root/autodl-tmp/new_dataset/3 Gaofen-1/'
ms_size=64
#测试设置
TIMESTAMP=datetime.datetime.now().strftime('%y-%m-%d-%H')
test_STAMP='23-03-15-20'
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
savedir = '/root/autodl-tmp/output/'
# train
train_dir = '/root/autodl-tmp/train/'
train_type = 'train_low_res'# full是无监督 low是有监督
data_type = "tanh"

scale_factor = 4
num_epochs = 120
cuda = True
device = torch.device("cuda:0" )
device_ids = [0]
parallel = True
make_data = True
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

