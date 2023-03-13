#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
from mmcv import Config#MMCv的核心组件 config类
from mmcv.utils import get_logger
import torch.utils.data as Data
import random
from function.utilis import *
from function.data_utils import *
from tensorboardX import SummaryWriter
# from pangan import generator2
from models.model_pangan  import discriminator
from models.model_DIGAN import P3Net
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from function.loss import D_loss,G_loss_adv
import torch.multiprocessing
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# torch.cuda.set_device(2,3)
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
    # model = Pangan()
    model_cfg = cfg.get('model_cfg', dict())
    G = P3Net()
    D_spatial = discriminator(in_channel=1)
    D_spectral = discriminator(in_channel=4)

    D_loss_func = D_loss()
    G_loss_func = G_loss_adv()

    # set GPU
    if cfg.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
    logger.info("===> Setting GPU")
    if cfg.cuda:
        logger.info('cuda_mode:', cfg.cuda)
        G.to(cfg.device)
        G_loss_func.to(cfg.device)
        D_loss_func.to(cfg.device)
        D_spectral.to(cfg.device)
        D_spatial.to(cfg.device)
        if cfg.parallel:
            G = nn.DataParallel(G, cfg.device_ids)
            D_spatial = nn.DataParallel(D_spatial, cfg.device_ids)
            D_spectral = nn.DataParallel(D_spectral, cfg.device_ids)
        num_params = 0
        for param in G.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
        num_params = 0
        for param in D_spatial.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
        logger.info("===> structure of spatial D")
        logger.info(D_spatial)
        logger.info("===> structure of spectral D")
        logger.info(D_spectral)
        logger.info("===> structure of generator")
        logger.info(G)
    # optimizer
    logger.info("===> Setting Optimizer")
    optim_D_spatial = torch.optim.RMSprop(D_spatial.parameters(), lr=cfg.lr)
    optim_D_spectral = torch.optim.RMSprop(D_spectral.parameters(), lr=cfg.lr)
    optim_G = torch.optim.RMSprop(G.parameters(), lr=cfg.lr)
    if cfg.pretrained and cfg.test:
        logger.info('==>loading test data...')
        make_data.generate_data(cfg.make_data_cfg['test_data']) #前面是保存位置
        test_dataset = TestDatasetFromFolder(os.path.join(cfg.train_dir, cfg.dataset,cfg.test_type), cfg.bit_depth, cfg.test_type,data_type='tanh')
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
        if cfg.make_data:
             #前面是保存位置
            make_data.generate_data(cfg.make_data_cfg['train_data'])  # 前面是保存位置
            make_data.generate_data(cfg.make_data_cfg['test_data'])
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
                gen_checkpoint_D1 = torch.load(cfg.resumeD_spatial)
                gen_checkpoint_D2 = torch.load(cfg.resumeD_spectral)

                cfg.start_epoch = gen_checkpoint_G['epoch'] + 1
                logger.info('==>start training at epoch {}'.format(cfg.start_epoch))
                G.load_state_dict(gen_checkpoint_G['model'].state_dict())
                D_spatial.load_state_dict(gen_checkpoint_D1['model'].state_dict())
                D_spectral.load_state_dict(gen_checkpoint_D2['model'].state_dict())
                logger.info("===> resume Training...")
                train(train_dataloader,test_dataloader,G,D_spatial,D_spectral, optim_D_spatial,optim_D_spectral,optim_G,D_loss_func,G_loss_func,cfg,logger)
            else:
                logger.info('==> cannot start training at epoch {}'.format(cfg.start_epoch))
        else:
            train(train_dataloader,test_dataloader,G,D_spatial,D_spectral, optim_D_spatial,optim_D_spectral,optim_G,D_loss_func,G_loss_func,cfg,logger)

# training
def train(train_dataloader,test_dataloader,
          G,D_spatial,D_spectral,
          optim_D_spatial,optim_D_spectral,optim_G,
          D_loss_func,G_loss_func, cfg ,logger):
    logger.info('==>Training...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs + 1):
        train_process(train_dataloader, test_dataloader, G, D_spatial, D_spectral, optim_D_spatial, optim_D_spectral,
                      optim_G, D_loss_func, G_loss_func, epoch, cfg,logger)

        save_checkpoint(G, 'G', cfg.dataset, epoch)
        save_checkpoint(D_spatial, 'D_spatial', cfg.dataset, epoch)
        save_checkpoint(D_spectral, 'D_spectral', cfg.dataset, epoch)
# testing
def test(test_dataloader, G, save_img_dir,cfg,info,epoch):
    logger.info('==>Testing...')
    test_process(test_dataloader, G, save_img_dir,cfg.test_type,cfg,info,epoch)
# train every epoch
def train_process(dataloader, test_dataloader,
                  G,D_spatial,D_spectral,
                  optim_D_spatial,optim_D_spectral,optim_G ,
                  D_loss_func,G_loss_func,
                  epoch,cfg,logger):
    lr = adjust_learning_rate(epoch-1,cfg.lr,cfg.step)
    for param_group in optim_D_spatial.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optim_D_spatial.param_groups[0]["lr"])
    for param_group in optim_D_spatial.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optim_D_spectral.param_groups[0]["lr"])
    for param_group in optim_G.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optim_G.param_groups[0]["lr"])
    writer = SummaryWriter(cfg.tensorboard_path)
    eval_images = []
    for iteration, batch in enumerate(dataloader):
        input_lr, input_pan, input_lr_up, target = batch[0], batch[1], batch[2], batch[3]
        G.train()
        D_spatial.train()
        D_spectral.train()
        if cfg.cuda:
            input_lr = input_lr.to(cfg.device)
            input_pan = input_pan.to(cfg.device)
            input_lr_up = input_lr_up.to(cfg.device)
            target = target.to(cfg.device)
        # -----------------------------------------------
        # training model
        # ------------------------------------------------
        for i in range(2):
            output,fake_pan,_= G(input_lr_up,input_lr, input_pan)
            # compute loss
            spatial_pos = D_spatial(high_pass(input_pan))
            spatial_neg = D_spatial(high_pass(fake_pan))

            spectral_pos = D_spectral(input_lr_up)
            spectral_neg = D_spectral(output)
            spatial_D_loss, spectral_D_loss = D_loss_func(spatial_pos, spatial_neg, spectral_pos, spectral_neg)
            D_loss = spatial_D_loss+spectral_D_loss
            optim_D_spatial.zero_grad()
            spatial_D_loss.backward(retain_graph=True)
            optim_D_spatial.step()

            optim_D_spectral.zero_grad()
            spectral_D_loss.backward()
            optim_D_spectral.step()
        output,fake_pan,G_loss_rec= G(input_lr_up,input_lr, input_pan)
        # compute loss
        spatial_neg = D_spatial(high_pass(fake_pan))
        spectral_neg = D_spectral(output)
        g_loss_adv = G_loss_func(spatial_neg,spectral_neg)
        G_loss = g_loss_adv+G_loss_rec
        if cfg.parallel:
            G_loss = G_loss.mean()
        optim_G.zero_grad()
        G_loss.backward()
        optim_G.step()
        #valid data result
        with torch.no_grad():
            if iteration % 50 == 0:
                G_loss_valid = 0
                D_loss_valid = 0
                for idx1, batch1 in enumerate(test_dataloader):
                    input_lr_eval, input_pan_eval, input_lr_up_eval,target_eval = batch1[0], batch1[1], batch1[2],batch1[3]  # 这里的index指的是在原始数数据集中的位置 便于找到测试图片
                    if cfg.cuda:
                        input_pan = input_pan_eval.to(cfg.device)
                        input_lr_up = input_lr_up_eval.to(cfg.device)
                        input_lr = input_lr_eval.to(cfg.device)
                                         # -----------------------------------------------
                    # evaluating model
                    # ------------------------------------------------
                    pansharpening, fake_pan,G_loss_rec = G(input_lr_up,input_lr, input_pan)
                    # compute loss
                    spatial_neg = D_spatial(high_pass(fake_pan))
                    spectral_neg = D_spectral(pansharpening)
                    g_loss_adv = G_loss_func(spatial_neg, spectral_neg)
                    G_loss_valid = g_loss_adv + G_loss_rec
                    if cfg.parallel:
                        G_loss_valid = G_loss_valid.mean()
                    # compute QNR
                    eval_compute(input_pan,input_lr,pansharpening,target,cfg.test_type,cfg.data_type,logger)
                    visual_img = visualize_img(torch2np(pansharpening)[3],'tanh')
                    zoom_img = zoom_in(visual_img)
                    plt.imshow(zoom_img)
                    plt.show()
                    eval_images.append(zoom_img)
                    #tiff_save_img([3], os.path.join('validation',img_name), 11, data_type='tanh')
                    #记录在tensorboard中
                    writer.add_scalar('D_loss',
                                  D_loss ,
                                      (epoch-1) * len(dataloader) + iteration)
                    writer.add_scalar('G_loss',
                                      G_loss,
                                      (epoch-1)* len(dataloader) + iteration)
                    writer.add_scalar('G_loss_valid',
                                      G_loss_valid,
                                      (epoch - 1) * len(dataloader) + iteration)
            logger.info('epoch:[{}/{}] batch:[{}/{}] D_loss:{:.5f}  G_loss:{:.5f}  eval_G_loss:{:.5f} '
                 .format(epoch, cfg.num_epochs, iteration, len(dataloader), D_loss, G_loss,G_loss_valid))
    with torch.no_grad():
        eval_images = np.array(eval_images).transpose(0, 3,1,2)
        print(eval_images.shape)
        save_name = 'validation/' + 'epoch_%d.png' %(epoch)
        save_image(torch.Tensor(eval_images), save_name, nrow=4, padding=2, pad_value=0,normalize=True,range = (0,255)) # 3*1024*1024*3 -> 3*3*1024
        img = cv2.imread(save_name)
        writer.add_image("img",img,global_step=epoch,dataformats='HWC') #HWC还是CHW要选好
        #plot show的参数需要 M N 3
        path = cfg.savedir
        auto_create_path(path)
        test(test_dataloader, G, path,cfg,logger,epoch)

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
        prediction,_,_ = G(input_lr_up,input_lr, input_pan)
        #out = image_clip(out, 0, 1)
        # target = image_clip(target, 0, 1)
        eval_compute(input_pan,input_lr,prediction,target,test_type,cfg.data_type,logger)
       #建立保存生成图片的文件夹
        save_GT_dir = os.path.join(save_img_dir, 'GT')
        save_pre_dir = os.path.join(save_img_dir, 'prediction')
        save_pan_dir = os.path.join(save_img_dir, 'pan')
        save_ms_dir = os.path.join(save_img_dir, 'ms')
        dir = [save_GT_dir,save_pre_dir,save_pan_dir,save_ms_dir]
        for son_dir in dir:
            if not os.path.exists(son_dir):
                os.mkdir(son_dir)
        out = torch2np(prediction)
        for i in range(out.shape[0]):
            # print(torch2np(out[i].unsqueeze(0)).shape)
            img_name = str(image_index[i]) + '.tif'
            tiff_save_img(out[i], os.path.join(save_pre_dir, img_name), cfg.bit_depth,
                          data_type='tanh')  # 先转换成numpy 再保存RGB
            tiff_save_img(torch2np(input_lr)[i], os.path.join(save_ms_dir, img_name), cfg.bit_depth,
                          data_type='tanh')  # 先转换成numpy 再保存RGB
            tiff_save_img(torch2np(input_pan)[i], os.path.join(save_pan_dir, img_name), cfg.bit_depth,
                          data_type='tanh')  # 先转换成numpy 再保存RGB


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
