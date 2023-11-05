import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import sys
if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    # root_dir='./output/QB_small/test_full_res'
    # Dict_method={
        
    #         # 'FusionNet':'130',
    #         #  'LACNET':'120',
    #         #  'Panformer':'130',
    #         #  'TFNet':'100',
    #         #  'MTF_GLP':'0',
    #         #  'Wavelet':'0',
    #         #  'IHS':'0',
    #          'GFPCA':'0',
    #         #  'MSDCNN_model':'130',
    #         #  'my_model_3_31_2':'130',
    #         #   'pan':'100',
    #         #    'ms':'100'
             
    #          }
    # dataset='QB_small/'
    # num='2'
    
    # root_dir='./output/WV2_small/test_full_res'
    # Dict_method={
        
    #         'FusionNet':'130',
    #          'LACNET':'120',
    #          'Panformer':'130',
    #          'TFNet':'70',
    #          'MTF_GLP':'0',
    #          'Wavelet':'0',
    #          'IHS':'0',
    #          'PCA':'0',
    #          'MSDCNN_model_8c':'160',
    #          'my_model_3_31_2':'100',
    #          'pan':'70',
    #          'ms':'70'
             
    #          }
    # dataset='WV2_small/'
    # num='1'
    
    root_dir='./output/WV4_small/test_full_res'
    Dict_method={
        
            # 'FusionNet':'130',
            #  'LACNET':'120',
            #  'Panformer':'130',
            #  'TFNet':'100',
            #  'MTF_GLP':'0',
            #  'Wavelet':'0',
            #  'IHS':'0',
            #  'GFPCA':'0',
            #  'MSDCNN_model':'130',
             'my_model_3_31_2':'140',
             'LDP_Net':'30',
               'pan':'100',
               'ms':'100'
             
             }
    dataset='WV4_small/'
    
    num='2'
    for key,value in Dict_method.items():
        if key=='pan':
            img=cv.imread(os.path.join(root_dir,'TFNet',value,'pan',str(num)+'.tif'))
        elif key=='ms':
            img=cv.imread(os.path.join(root_dir,'TFNet',value,'ms',str(num)+'.tif'))
            img=cv.resize(img,(1024,1024), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        else:
            img = cv.imread(os.path.join(root_dir,key,value,'prediction',str(num)+'.tif'))
        
        img_width = img.shape[0]
        img_height = img.shape[1]
        scale = 3
        # # QB FR
        # x_start = int(0.38 * img_width)
        # y_start = int(0.50 * img_width)
        # mask_width = int(0.15*img_width)
       
        # # WV2 FR
        # x_start = int(0.32 * img_width)
        # y_start = int(0.4 * img_width)
        # mask_width = int(0.15*img_width)
        
        #WV4 FR
        x_start = int(0.6 * img_width)
        y_start = int(0.3 * img_width)
        mask_width = int(0.15*img_width)
        mask_start_x =img_width-mask_width*scale
        mask_start_y =img_width-mask_width*scale


        part = img[x_start:x_start+mask_width, y_start:y_start+mask_width]
        # 双线性插值法
        mask = cv.resize(part, (scale*mask_width, scale*mask_width), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        if img is None is None:
            print('Failed to read picture')
            sys.exit()

        # 放大后局部图的位置img[210:410,670:870]
        img[mask_start_x:mask_start_x+scale*mask_width, mask_start_y:mask_start_y+scale*mask_width] = mask

        # 画框并连线
 
        cv.rectangle(img, (y_start, x_start), (y_start+mask_width, x_start+mask_width), (0, 0, 255), 10)
        cv.rectangle(img, (mask_start_y, mask_start_x), (mask_start_y+scale*mask_width, mask_start_x+scale*mask_width),  (0, 0, 255), 16)   
       
        # 展示结果
       # cv.imshow('img', img)
        save_dir='./img_result/'+dataset+num+'/multi2/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_name = os.path.join(os.path.join(save_dir,key+'_predict.jpg'))
        cv.imwrite(img_name, img)
        fig4=plt.figure(dpi=300,figsize=(15,15))
   
    # cv.waitKey(0)
    # cv.destroyAllWindows()
