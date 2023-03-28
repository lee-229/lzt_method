import numpy as np
import os
import matplotlib.pyplot as plt
from data_utils import load_image
x = np.arange(0, 256)
y = np.arange(0, 256)
dataset='QB_small'
img_path='/media/dy113/disk1/Project_lzt/code/lzt_method/output/'+dataset+'/test_low_res'
Dict_method={'LACNET':'120',
            #  'PanNet':'110',
             'Panformer':'130',
             'TFNet':'100',
            #  'PanNet':'150',
             'my_model_3_13':'130',
             }
# Dict_method={'LACNET':'120',
#              'PanNet':'150',
#              'Panformer':'130',
#              'TFNet':'100',
#              'PanNet':'70',
#              'my_model_3_13':'130',
#              }
num=30
test_dir='./img_result/'+dataset
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
fig=plt.figure(figsize=(10,10))
levels=np.arange(0,100,1)  
for key,value in Dict_method.items():
    predict = load_image(os.path.join(img_path,key,value,'prediction',str(num)+'.tif'))
    GT=load_image(os.path.join(img_path,key,value,'GT',str(num)+'.tif'))
    residual=GT-predict
    plt.contourf(np.abs(residual)[1,:,:],cmap = 'jet',levels = levels)
    plt.axis('off')
    save_path=os.path.join(test_dir,str(num))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,key+'.jpg'),bbox_inches='tight',pad_inches = 0.0)   

# cb = plt.colorbar()

# plt.show()




