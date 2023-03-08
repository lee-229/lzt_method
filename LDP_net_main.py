#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import torch
from mmcv import Config#MMCv的核心组件 config类
from models.LDP_net import LDP_Net
import argparse
import torch.utils.data as Data
from mmcv.utils import get_logger
import random
from function.utilis import *
from function.data_utils import *
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# training settings

def parse_args():
    parser = argparse.ArgumentParser(description='pan-sharpening implementation')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    return parser.parse_args()


def main(cfg, logger):
    seed = random.randint(1, 10000)
    logger.info(seed)
    if cfg.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    # build network
    print('==>building network...')
    G = LDP_Net(in_channel=cfg.in_nc, mid_channel=cfg.mid_nc)
    num_params = 0
    for param in G.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
    # loss
    pixel_loss = torch.nn.MSELoss()
    if cfg.pixel_loss_type == 'L1':
        pixel_loss = torch.nn.L1Loss()
    elif cfg.pixel_loss_type == 'L2':
        pixel_loss = torch.nn.MSELoss()
    kl_loss = torch.nn.KLDivLoss(reduction='sum')
    Smooth_operator = Smooth(in_nc=cfg.in_nc)
    # set GPU
    if cfg.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
    logger.info("===> Setting GPU")
    if cfg.cuda:
        logger.info('cuda_mode:', cfg.cuda)
        G.to(cfg.device)
        pixel_loss.to(cfg.device)
        kl_loss.to(cfg.device)
        Smooth_operator.to(cfg.device)
        if cfg.parallel:
            G = nn.DataParallel(G, cfg.device_ids)
    # optimizer
    print("===> Setting Optimizer")
    optim = torch.optim.Adam(G.parameters(), lr=cfg.lr)
    if cfg.pretrained and cfg.test:
        logger.info('==>loading test data...')
        #make_data.generate_data(cfg.make_data_cfg['test_data']) #前面是保存位置
        test_dataset = TestDatasetFromFolder(os.path.join(cfg.train_dir, cfg.dataset,cfg.test_type), cfg.bit_depth, cfg.test_type,cfg.data_type)
        test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
                                          num_workers=cfg.threads)
        if os.path.isfile(cfg.pretrained):
            logger.info('==> loading model {}'.format(cfg.pretrained))
            model_weights = torch.load(cfg.pretrained)
            G.load_state_dict(model_weights['model'].state_dict())
            test(test_dataloader, G, cfg.savedir,cfg)
        else:
            logger.info('==> no model found at {}'.format(cfg.pretrained))

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
                gen_checkpoint_G = torch.load(cfg.resumeG)
                cfg.start_epoch = gen_checkpoint_G['epoch'] + 1
                logger.info('==>start training at epoch {}'.format(cfg.start_epoch))
                G.load_state_dict(gen_checkpoint_G['model'].state_dict())

                logger.info("===> resume Training...")
                train(train_dataloader,test_dataloader, G,optim, kl_loss, pixel_loss,Smooth_operator,cfg,logger)
            else:
                logger.info('==> cannot start training at epoch {}'.format(cfg.start_epoch))
        else:
            train(train_dataloader,test_dataloader, G,optim, kl_loss, pixel_loss,Smooth_operator,cfg,logger)

# training
def train(train_dataloader,test_dataloader, model,optim, kl_loss, pixel_loss,Smooth_operator,cfg,logger):
    logger.info('==>Training...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs + 1):
        train_process(train_dataloader, test_dataloader,
                      model, optim,
                  kl_loss, pixel_loss,Smooth_operator,
                  epoch,cfg,logger)
        if epoch % 10 == 0:
            save_checkpoint(model, 'G', cfg.dataset,cfg.model, epoch)
# train every epoch
def train_process(dataloader, test_dataloader,
                  G,optim_G ,
                  kl_loss, pixel_loss,Smooth_operator,
                  epoch,cfg,logger):
    lr = adjust_learning_rate(epoch - 1, cfg.lr, cfg.step)
    for param_group in optim_G.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optim_G.param_groups[0]["lr"])
    losses = []
    for iteration, batch in enumerate(dataloader):
        input_lr, input_pan, input_lr_up, target = batch[0], batch[1], batch[2], batch[3]
        G.train()

        if cfg.cuda:
            input_lr = input_lr.to(cfg.device)
            input_pan = input_pan.to(cfg.device)
            input_lr_up = input_lr_up.to(cfg.device)
        # -----------------------------------------------
        # training model
        # ------------------------------------------------
        pan_multi = stack(input_pan, r=cfg.in_nc)
        ms, lr_ms, gray_ms, lr_pan, lrms_up_gray = G(input_lr_up, pan_multi)
        optim_G.zero_grad()

        # compute loss
        # spectral loss
        # spectral_low
        ms_smooth = Smooth_operator(ms)
        ms_ = F.interpolate(ms_smooth, [32, 32], mode='bilinear', align_corners=True)#???为什么要插值 为了和input_lr一样小 然后计算loss
        loss_ = 20 * pixel_loss(ms_, input_lr)
        # spectral_high
        loss_lr_ms = pixel_loss(lr_ms, input_lr_up)
        loss_spectral = loss_ + loss_lr_ms
        # spatial loss
        # spatial_high
        loss_ms_gray = pixel_loss(gray_ms, pan_multi)
        # spatial_low
        loss_lr_pan = pixel_loss(lr_pan, lrms_up_gray)
        loss_spatial = 20 * loss_ms_gray + loss_lr_pan
        # KL loss
        res1 = input_lr_up - lrms_up_gray
        res2 = ms - pan_multi
        loss_kl = 0.1 * kl_loss(res1.softmax(dim=-1).log(), res2.softmax(dim=-1))
        # total loss
        loss = 5*loss_spatial + 5 * loss_spectral + loss_kl
        losses.append(loss.item())
        loss.backward()
        optim_G.step()
        logger.info('epoch:[{}/{}] batch:[{}/{}]  G_loss:{:.5f}   '
                    .format(epoch, cfg.num_epochs, iteration, len(dataloader),loss))

# testing
def test(test_dataloader, G, save_img_dir,cfg,logger,epoch):
    logger.info('==>Testing...')
    save_img_dir = os.path.join(save_img_dir, cfg.dataset, cfg.test_type, cfg.model, str(epoch))
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
        pan_multi = stack(input_pan, r=cfg.in_nc)
        prediction,_,_,_,_ = G(input_lr_up, pan_multi)
        prediction = image_clip(prediction, 0, 1)
        # target = image_clip(target, 0, 1)
        eval_compute(input_pan, input_lr, prediction, target, cfg.test_type, cfg.data_type,logger)
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
                          data_type='sigmoid')  # 先转换成numpy 再保存RGB
            tiff_save_img(torch2np(input_lr)[i], os.path.join(save_ms_dir, img_name), cfg.bit_depth,
                          data_type='sigmoid')  # 先转换成numpy 再保存RGB
            tiff_save_img(torch2np(input_pan)[i], os.path.join(save_pan_dir, img_name), cfg.bit_depth,
                          data_type='sigmoid')  # 先转换成numpy 再保存RGB

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
