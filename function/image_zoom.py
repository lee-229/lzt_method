import os

import cv2 as cv
import sys
if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    dir='GF-1/'
    for x in os.listdir(dir):
        img = cv.imread(dir+x)
        if  x=='ms.png':
            img = cv.resize(img, (1024,1024), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        img_width = img.shape[0]
        img_height = img.shape[1]
        # 需要放大的部分
        # WV4 0.2 0.2 0.2  QB 0.4 0.45 0.15
        x_start = int(0.3 * img_width)
        y_start = int(0.3 * img_width)
        mask_width = int(0.15*img_width)
        scale = 3
        mask_start_x =img_width-mask_width*scale
        mask_start_y =0

        part = img[x_start:x_start+mask_width, y_start:y_start+mask_width]
        # 双线性插值法
        mask = cv.resize(part, (scale*mask_width, scale*mask_width), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        if img is None is None:
            print('Failed to read picture')
            sys.exit()

        # 放大后局部图的位置img[210:410,670:870]
        img[mask_start_x:mask_start_x+scale*mask_width, mask_start_y:mask_start_y+scale*mask_width] = mask

        # 画框并连线
        cv.rectangle(img, (y_start, x_start), (y_start+mask_width, x_start+mask_width), (0, 255, 0), 4//4)
        cv.rectangle(img, (mask_start_y, mask_start_x), (mask_start_y+scale*mask_width, mask_start_x+scale*mask_width), (0, 255, 0), 4//4)
        # img = cv.line(img, (350, 300), (570, 110), (0, 255, 0))
        # img = cv.line(img, (350, 400), (570, 410), (0, 255, 0))
        # 展示结果
       # cv.imshow('img', img)
        img_name = os.path.join(dir,'multi_'+x)
        cv.imwrite(img_name, img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
