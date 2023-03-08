#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os

from mmcv import Config#MMCv的核心组件 config类
from mmcv.utils import get_logger
import torch.utils.data as Data
import random
from function.utilis import *
from function.data_utils import *
from tensorboardX import SummaryWriter
# from pangan import generator2
from models.model_Unet import Unet_cutblock
from models.model_pangan import discriminator
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from function.loss import G_loss_unet_adv,D_loss_unet,super_loss
from collections import OrderedDict
import csv

import torch.multiprocessing

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def parse_args():
    parser = argparse.ArgumentParser(description='pan-sharpening implementation')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    return parser.parse_args()


def main(cfg, logger):
    seed = random.randint(1, 10000)
    logger.info("Random Seed: ", seed)
    torch.manual_seed(seed)
    if cfg.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    # build network
    logger.info('==>building network...')
    #定义生成器和鉴别器
    G = Unet_cutblock()
    D = discriminator(4)
    #损失函数
    G_loss_adv_func = G_loss_unet_adv()
    G_loss_rec_func = super_loss()
    D_loss_func = D_loss_unet()

    # set GPU
    if cfg.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
    logger.info("===> Setting GPU")
    if cfg.cuda:
        logger.info('cuda_mode:', cfg.cuda)
        G.to(cfg.device)
        D.to(cfg.device)
        G_loss_adv_func = G_loss_adv_func.to(cfg.device)
        G_loss_rec_func = G_loss_rec_func.to(cfg.device)
        D_loss_func = D_loss_func.to(cfg.device)
        if cfg.parallel:
            G = nn.DataParallel(G,cfg.device_ids)
            D = nn.DataParallel(D, cfg.device_ids)
        num_params = 0
        for param in G.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))

        logger.info("===> structure of generator")
        logger.info(G)

    # optimizer
    logger.info("===> Setting Optimizer")
    optim_G = torch.optim.RMSprop(G.parameters(), lr=cfg.lr)
    optim_D = torch.optim.RMSprop(D.parameters(), lr=cfg.lr)
    # # 产生测试数据
    # # 生成全尺度测试数据
    # make_data.generate_data(cfg.make_data_cfg['test_data'])  # 产生测试数据 cfg给定测试数据源 测试数据对数 裁剪步幅 以及裁剪图片大小等
    test_dataset = TestDatasetFromFolder(cfg.train_set_cfg['dataset_test'])  # 产生dataloader train_set_cfg给定读取测试图片的位置（full low)(何种数据集）
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
                                      num_workers=cfg.threads)
    # # 生成降尺度测试数据
    make_data.generate_data(cfg.make_data_cfg['test_data_2'])
    test_dataset_2 = TestDatasetFromFolder(cfg.train_set_cfg['dataset_test_2'])
    test_dataloader_2 = Data.DataLoader(dataset=test_dataset_2, batch_size=cfg.test_batch_size, shuffle=False,
                                        num_workers=cfg.threads)
    # 产生训练数据
    if cfg.make_data:
        make_data.generate_data(cfg.make_data_cfg['train_data'])  # 前面是保存位置
    if cfg.pretrained and cfg.test:
        logger.info('==>loading test data...')
        if os.path.isfile(cfg.pretrained):
            logger.info('==> loading model {}'.format(cfg.pretrained))
            model_weights = torch.load(cfg.pretrained)
            G.load_state_dict(model_weights)
            test(test_dataloader, G, cfg.savedir, cfg.test_type, cfg, logger)
            test(test_dataloader_2, G, cfg.savedir, cfg.test_type_2, cfg, logger)

            with open(cfg.csv_FR_dir) as csv_file:
                row = csv.reader(csv_file, delimiter=',')
                next(row)  # 读取首行
                D_lambda = []  # 建立一个数组来存储股价数据
                D_s = []
                QNR = []
                # 读取除首行之后每一行的第二列数据，并将其加入到数组price之中
                for r in row:
                    D_lambda.append(float(r[1]))  # 将字符串数据转化为浮点型加入到数组之中
                    D_s.append(float(r[2]))  # 将字符串数据转化为浮点型加入到数组之中
                    QNR.append(float(r[3]))  # 将字符串数据转化为浮点型加入到数组之中
                print('D_lambda', round(np.mean(D_lambda),4), round(np.var(D_lambda),4))
                print('D_s', round(np.mean(D_s),4), round(np.var(D_s),4))
                print('QNR', round(np.mean(QNR),4), round(np.var(QNR),4))

            with open(cfg.csv_RR_dir) as csv_file:
                row = csv.reader(csv_file, delimiter=',')
                next(row)  # 读取首行
                SAM = []  # 建立一个数组来存储股价数据
                ERGAS = []
                Q4 = []
                # 读取除首行之后每一行的第二列数据，并将其加入到数组price之中
                for r in row:
                    SAM.append(float(r[1]))  # 将字符串数据转化为浮点型加入到数组之中
                    ERGAS.append(float(r[2]))  # 将字符串数据转化为浮点型加入到数组之中
                    Q4.append(float(r[3]))  # 将字符串数据转化为浮点型加入到数组之中
                print('SAM', round(np.mean(SAM),4), round(np.var(SAM), 4))
                print('ERGAS', round(np.mean(ERGAS),4), round(np.var(ERGAS),4))
                print('Q4', round(np.mean(Q4),4),round(np.var(Q4),4))

                # train every epoch
    else:
        logger.info('==>loading training data...')
        logger.info(cfg.train_set_cfg['dataset_train'])
        train_dataset = TrainDatasetFromFolder(cfg.train_set_cfg['dataset_train'])
        train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=False,
                                           num_workers=cfg.threads)
        test_dataset = TestDatasetFromFolder(cfg.train_set_cfg['dataset_test'])
        test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
                                          num_workers=cfg.threads)
        if cfg.resumeG:
            if os.path.isfile(cfg.resumeG):
                model_weights_G = torch.load(cfg.resumeG)
                G.load_state_dict(model_weights_G)
                model_weights_D = torch.load(cfg.resumeD)
                G.load_state_dict(model_weights_G)
                logger.info("===> resume Training...")
                train(train_dataloader,
                      G, D, optim_G, optim_D, G_loss_rec_func, G_loss_adv_func, D_loss_func,
                      cfg, logger)
            else:
                logger.info('==> cannot start training at epoch {}'.format(cfg.start_epoch))
        else:
            train(train_dataloader,
                  G, D, optim_G, optim_D, G_loss_rec_func,G_loss_adv_func, D_loss_func,
                  cfg, logger)

# training
def train(train_dataloader,
          G, D, optim_G, optim_D, G_loss_rec_func,G_loss_adv_func, D_loss_func,
          cfg, logger):
    logger.info('==>Training...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs + 1):
        train_process(train_dataloader,
                  G, D, optim_G, optim_D,G_loss_rec_func,G_loss_adv_func, D_loss_func,
                  epoch,cfg, logger)
        if epoch%10==0:
            save_checkpoint(G, 'G', cfg.dataset, cfg.model,epoch)

# testingf
def test(test_dataloader, G, save_img_dir,test_type,cfg,logger):
    logger.info('==>Testing...')
    save_img_dir = os.path.join(save_img_dir, cfg.dataset, test_type, cfg.model, str(cfg.num_epochs))
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    # G.eval()

    for idx, batch in enumerate(test_dataloader):
        input_lr, input_pan, input_lr_up, target, image_index = batch[0], batch[1], batch[2], batch[3], batch[
            4]  # 这里的index指的是在原始数数据集中的位置 便于找到测试图片
        input_pan = input_pan.to(cfg.device)
        input_lr = input_lr.to(cfg.device)
        input_lr_up = input_lr_up.to(cfg.device)
        target = target.to(cfg.device)
        ##不同的模型生成表达式不同
        #TFnet
        prediction= G(input_pan,input_lr_up)
        # LDPnet
        # pan_multi = stack(input_pan, r=4)
        # prediction, _, _, _, _ = G(input_lr_up, pan_multi)

        # out = image_clip(out, 0, 1)
        # target = image_clip(target, 0, 1)

        #写表头
        if test_type == 'test_low_res':
            header = ['image_index', 'SAM', 'ERGAS', 'Q4', 'SCC']
        else:
            header = ['image_index', 'D_lambda', 'D_s', 'QNR','SF','FCC']
            #返回字典
        results = eval_compute(input_pan, input_lr, prediction, target,test_type,cfg.data_type, logger)
        if test_type == 'test_low_res':
            with open(cfg.csv_RR_dir, 'a', encoding='utf-8', newline='') as file_obj:
                Writer = csv.writer(file_obj, header)
                if idx==0:
                    Writer.writerow(header)
                Writer.writerow([image_index[0], np.array(results['SAM'][0]), np.array(results['ERGAS'][0]), np.array(results['Q4'][0]),np.array(results['SCC'][0])])

        else:
            with open(cfg.csv_FR_dir, 'a', encoding='utf-8', newline='') as file_obj:
                Writer = csv.writer(file_obj, header)
                if idx==0:
                    Writer.writerow(header)
                Writer.writerow([image_index[0],np.array(results['D_lambda'][0]),np.array(results['D_s'][0]),np.array(results['QNR'][0]),np.array(results['SF'][0]),np.array(results['FCC'][0])])
        # 建立保存生成图片的文件夹
        save_GT_dir = os.path.join(save_img_dir, 'GT')
        save_pre_dir = os.path.join(save_img_dir, 'prediction')
        save_pan_dir = os.path.join(save_img_dir, 'pan')
        save_ms_dir = os.path.join(save_img_dir, 'ms')
        dir = [save_GT_dir, save_pre_dir, save_pan_dir, save_ms_dir]
        for son_dir in dir:
            if not os.path.exists(son_dir):
                os.mkdir(son_dir)
        out = torch2np(prediction)
        for i in range(out.shape[0]):
            # print(torch2np(out[i].unsqueeze(0)).shape)
            img_name = str(image_index[i]) + '.tif'
            tiff_save_img(out[i], os.path.join(save_pre_dir, img_name), cfg.bit_depth,
                          data_type=cfg.data_type)  # 先转换成numpy 再保存RGB
            #tiff_save_img_no_his(out[i], os.path.join(save_pre_dir, str(image_index[i]) + '_no_his'+'.tif'), cfg.bit_depth,
                         # data_type=cfg.data_type)  # 先转换成numpy 再保存RGB
            tiff_save_img(torch2np(input_lr)[i], os.path.join(save_ms_dir, img_name), cfg.bit_depth,
                          data_type=cfg.data_type)  # 先转换成numpy 再保存RGB
            tiff_save_img(torch2np(input_pan)[i], os.path.join(save_pan_dir, img_name), cfg.bit_depth,
                          data_type=cfg.data_type)  # 先转换成numpy 再保存RGB
            if test_type == 'test_low_res':
                tiff_save_img(torch2np(target)[i], os.path.join(save_GT_dir, img_name), cfg.bit_depth,
                              data_type=cfg.data_type)  # 先转换成numpy 再保存RGB
    #计算均值和方差


def train_process(train_dataloader,
                  G, D, optim_G, optim_D, G_loss_rec_func,G_loss_adv_func, D_loss_func,
                  epoch,cfg, logger):
    lr = adjust_learning_rate(epoch-1,cfg.lr,cfg.step)
    for param_group in optim_G.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optim_G.param_groups[0]["lr"])
    writer = SummaryWriter(cfg.tensorboard_path)
    for iteration, batch in enumerate(train_dataloader):
        input_lr, input_pan, input_lr_up, target = batch[0], batch[1], batch[2], batch[3]
        G.train()
        if cfg.cuda:
            input_lr = input_lr.to(cfg.device)
            input_pan = input_pan.to(cfg.device)
            input_lr_up = input_lr_up.to(cfg.device)
            target = target.to(cfg.device)
        # -----------------------------------------------
        # training model
        # ------------------------------------------------
        output = G(input_pan, input_lr_up)
        for i in range(1):
            # compute loss
            optim_D.zero_grad()
            pos = D(target)
            neg = D(output)
            D_loss = D_loss_func(pos,neg)
            D_loss.backward(retain_graph=True)#循环训练需要加入 retain_graph=True
            optim_D.step()
        # #WGAN专用
        # for p in D.parameters():
        #     p.data.clamp_(-opt.clip_value, opt.clip_value)
        #output = G(input_pan, input_lr_up)
        pos = D(target)
        neg = D(output)
        G_loss_rec = G_loss_rec_func(output,target)
        G_loss_adv = G_loss_adv_func(neg) #有监督学习
        G_loss = 0.005*G_loss_adv + G_loss_rec
        D_loss = D_loss_func(pos, neg)
        if cfg.parallel:
            G_loss = G_loss.mean()
        optim_G.zero_grad()
        G_loss.backward()
        optim_G.step()
        # 记录在tensorboard中
        writer.add_scalar('G_loss',
                          G_loss,
                          (epoch-1)* len(train_dataloader) + iteration)
        writer.add_scalar('G_adv_loss',
                          G_loss_adv,
                          (epoch - 1) * len(train_dataloader) + iteration)
        writer.add_scalar('D_loss',
                          D_loss,
                          (epoch - 1) * len(train_dataloader) + iteration)
        logger.info(
            'epoch:[{}/{}] batch:[{}/{}] D_loss:{:.5f} '.format(epoch, cfg.num_epochs, iteration, len(train_dataloader),D_loss))
        logger.info('epoch:[{}/{}] batch:[{}/{}] G_loss:{:.5f} G_loss_rec:{:.5f} g_loss_adv:{:.5f} '.format(epoch, cfg.num_epochs, iteration, len(train_dataloader),G_loss,G_loss_rec,G_loss_adv))
        path = cfg.savedir
        auto_create_path(path)

# testing code
def test_process(test_dataloader, G, save_img_dir,test_type,cfg,logger,epoch):
    save_img_dir = os.path.join(save_img_dir, cfg.dataset , cfg.test_type , cfg.model,str(epoch))
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    # G.eval()

    for idx, batch in enumerate(test_dataloader):
        input_lr,input_pan, input_lr_up, target,image_index= batch[0],batch[1], batch[2], batch[3] ,batch[4] #这里的index指的是在原始数数据集中的位置 便于找到测试图片
        input_pan = input_pan.to(cfg.device)
        input_lr = input_lr.to(cfg.device)
        input_lr_up = input_lr_up.to(cfg.device)
        target = target.to(cfg.device)
        prediction= G( input_pan,input_lr_up)
        if idx==0:
            pan = input_pan
            ms = input_lr
            predict = prediction
            tar = target
        else:
            pan = torch.cat([pan,input_pan],dim=0)
            ms = torch.cat([ms, input_lr], dim=0)
            predict = torch.cat([predict, prediction], dim=0)
            tar = torch.cat([tar, target], dim=0)

        input_pan = torch2np(input_pan)
        input_lr = torch2np(input_lr)
        out = torch2np(prediction)
        target = torch2np(target)

        save_GT_dir = os.path.join(save_img_dir, 'GT')
        save_pre_dir = os.path.join(save_img_dir, 'prediction')
        save_pan_dir = os.path.join(save_img_dir, 'pan')
        save_ms_dir = os.path.join(save_img_dir, 'ms')
        dir = [save_GT_dir, save_pre_dir, save_pan_dir, save_ms_dir]
        for son_dir in dir:
            if not os.path.exists(son_dir):
                os.mkdir(son_dir)

        for i in range(out.shape[0]):
            # print(torch2np(out[i].unsqueeze(0)).shape)
            img_name = str(image_index[i]) + '.tif'
            tiff_save_img(out[i], os.path.join(save_pre_dir, img_name), cfg.bit_depth,
                          data_type='tanh')  # 先转换成numpy 再保存RGB
            tiff_save_img(input_lr[i], os.path.join(save_ms_dir, img_name), cfg.bit_depth,
                          data_type='tanh')  # 先转换成numpy 再保存RGB
            tiff_save_img(input_pan[i], os.path.join(save_pan_dir, img_name), cfg.bit_depth,
                          data_type='tanh')  # 先转换成numpy 再保存RGB
            if test_type == 'test_low_res':
                tiff_save_img(target[i], os.path.join(save_GT_dir, img_name), cfg.bit_depth,
                              data_type=cfg.data_type)  # 先转换成numpy 再保存RGB
        #out = image_clip(out, 0, 1)
        # target = image_clip(target, 0, 1)

    eval_compute(pan,ms,predict,tar,test_type,cfg.data_type,logger)
       #建立保存生成图片的文件夹



# pretained
if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)  ##读取配置文件
    mmcv.mkdir_or_exist(cfg.log_dir)
    logger = get_logger('mmFusion', cfg.log_file, cfg.log_level)
    # logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info('===> Setting Random Seed')
    main(cfg, logger)
    # except:
    #     logger.error(str(traceback.format_exc()))
