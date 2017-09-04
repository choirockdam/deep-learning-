#coding:utf-8
"""
opencv边缘检测
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/white.jpg',0)
#使用16位有符号的数据类型,防止截断,dx,dy求导阶数
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
#用convertScaleAbs()函数将其转回原来的uint8形式
uint8_x = cv2.convertScaleAbs(x)
uint8_y = cv2.convertScaleAbs(y)
#sobel算子在两个方向上加权
sobel_img = cv2.addWeighted(uint8_x,0.5,uint8_y,0.5,0)
cv2.imshow('original img',img)
cv2.imshow('sobel img',sobel_img)

#laplacian算子,ksize是算子的大小，必须为1、3、5、7。默认为1
lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
#用convertScaleAbs()函数将其转回原来的uint8形式
laplacian = cv2.convertScaleAbs(lap)
cv2.imshow('laplacian img',laplacian)

#高斯滤波器对图像降噪,横向和纵向滤波系数为0
gaussianblur = cv2.GaussianBlur(img,(3,3),0)
#canny边缘检测,阈值1,2检测图像中明显的边缘
canny = cv2.Canny(gaussianblur,50,150)
cv2.imshow('canny img',canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
