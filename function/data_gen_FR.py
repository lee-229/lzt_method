import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from data_utils import load_image
x = np.arange(0, 256)
y = np.arange(0, 256)
dataset='WV2_small'
img_path='/media/dy113/disk1/Project_lzt/code/lzt_method/output/'+dataset+'/test_low_res'
Dict_method={
        
            'FusionNet':'130',
             'LACNET':'120',
             'Panformer':'130',
             'TFNet':'70',
             'MTF_GLP':'0',
             'Wavelet':'0',
             'SFIM':'0',
             'Brovey':'0',
             'MSDCNN_model_8c':'160',
             'my_model_3_31_2':'100',
             
             }
num=1



type1='residual'
type2='predict'
test_dir='./img_result/'+dataset
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for key,value in Dict_method.items():
    predict = load_image(os.path.join(img_path,key,value,'prediction',str(num)+'.tif'))
    save_path=os.path.join(test_dir,str(num))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig2=plt.figure(dpi=300,figsize=(40,40))
    plt.axis('off')
    plt.imshow(np.array(predict).transpose(1,2,0)/255)
    plt.savefig(os.path.join(save_path,key+'_predict.jpg'),bbox_inches='tight',pad_inches = 0.0)   
    
  
# cb = plt.colorbar()

# plt.show()




