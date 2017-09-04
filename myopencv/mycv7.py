#coding:utf-8
"""
opencv形态学处理
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/figure_6.png')
cv2.imshow('original image',img)
#定义3*3的十字形结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

#膨胀图像
dilated = cv2.dilate(img,kernel)
cv2.imshow('dilated image',dilated)

#腐蚀图像
eroded = cv2.erode(img,kernel)
cv2.imshow('eroded image',eroded)

#对图像进行开运算去除噪音
opened = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('open img',opened)

#对图像进行闭运算连接主要信息
closed = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closed img',closed)

cv2.waitKey(0)
cv2.destroyAllWindows()
