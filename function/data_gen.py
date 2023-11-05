import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei' # 黑体

from data_utils import load_image
x = np.arange(0, 256)
y = np.arange(0, 256)


# dataset='WV2_small'
# Dict_method={
#             'GSA':'0',
#             # 'FusionNet':'130',
#             #  'LACNET':'120',
#             #  'Panformer':'130',
#             #  'TFNet':'70',
#             #  'MTF_GLP':'0',
#             # 'Wavelet':'0',
#             # 'GFPCA':'0',
#             #    'IHS':'0',
#             #  'MSDCNN_model_8c':'160',
#             #  'my_model_3_31_2':'100',
             
#              }
# num=26

# dataset='WV4_small'
# Dict_method={
#             'GSA':'0',
#             # 'FusionNet':'130',
#             #  'LACNET':'120',
#             #  'Panformer':'130',
#             #  'TFNet':'100',
#             #  'MTF_GLP':'0',
#             # 'Wavelet':'0',
#             # 'GFPCA':'0',
#             #    'IHS':'0',
#             #  'MSDCNN_model':'130',
#             #  'my_model_3_31_2':'160',
             
#              }
# num=40

dataset='QB_small'
Dict_method={
            
            'GFPCA':'0',
            # 'FusionNet':'130',
            #  'LACNET':'120',
            #  'Panformer':'130',
            #  'TFNet':'100',
            #  'MTF_GLP':'0',
            #  'Wavelet':'0',
            #  'PCA':'0',
            #  'IHS':'0',
            #  'MSDCNN_model':'130',
            #  'my_model_3_31_2':'130',
             
             }
num=25
img_path='/media/dy113/disk1/Project_lzt/code/lzt_method/output/'+dataset+'/test_low_res'

type1='residual'
type2='predict'
test_dir='./img_result/'+dataset
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
fig=plt.figure(dpi=300,figsize=(10,10))
levels=np.arange(0,150,1)  
for key,value in Dict_method.items():
    predict = load_image(os.path.join(img_path,key,value,'prediction',str(num)+'.tif'))
    lr = load_image(os.path.join(img_path,key,value,'ms',str(num)+'.tif'))
    pan = load_image(os.path.join(img_path,key,value,'pan',str(num)+'.tif'))
    GT=load_image(os.path.join(img_path,key,value,'GT',str(num)+'.tif'))
    residual=GT-predict
   
    plt.contourf(np.abs(np.mean(residual,axis=0)),cmap = 'jet',levels = levels)
    plt.axis('off')
    save_path=os.path.join(test_dir,str(num))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # num_jpg = 100
    # predict=np.array(predict).transpose(1,2,0)/255
    
    
    #img.save(os.path.join(save_path,key+'_predict.jpg'),quality=100,subsampling=0)
    #Image.save(os.path.join(save_path,key+'_predict.jpg'), predict) 
    plt.savefig(os.path.join(save_path,key+'.jpg'),bbox_inches='tight',pad_inches = 0.0)   
    
    #img = Image.open(os.path.join(img_path,key,value,'prediction',str(num)+'.tif'))
    fig2=plt.figure(dpi=300,figsize=(10,10))
    plt.axis('off')
    plt.imshow(np.array(predict).transpose(1,2,0)/255)
    plt.savefig(os.path.join(save_path,key+'_predict.jpg'),bbox_inches='tight',pad_inches = 0.0)   
    
    if key=='GFPCA':
        fig3=plt.figure(dpi=300,figsize=(10,10))
        plt.axis('off')
        plt.imshow(np.array(predict).transpose(1,2,0)/255)
        plt.savefig(os.path.join(save_path,key+'_GT_predict.jpg'),bbox_inches='tight',pad_inches = 0.0)   
        fig4=plt.figure(dpi=300,figsize=(15,15))
        plt.axis('off')
        plt.imshow(np.array(lr).transpose(1,2,0)/255)
        plt.savefig(os.path.join(save_path,key+'_lr_predict.jpg'),bbox_inches='tight',pad_inches = 0.0)  
        fig5=plt.figure(dpi=300,figsize=(10,10))
        plt.axis('off')
        plt.imshow(np.array(pan)/255,cmap = plt.get_cmap('gray'))
        plt.savefig(os.path.join(save_path,key+'_pan_predict.jpg'),bbox_inches='tight',pad_inches = 0.0)    





