#coding:utf-8
"""
opencv形态学边缘检测
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/figure_2.png',0)
cv2.imshow('original img',img)

#定义一个3*3的十字形结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

#图像膨胀
dilated = cv2.dilate(img,kernel)
#图像腐蚀
eroded = cv2.erode(img,kernel)
#边缘检测
edge = cv2.absdiff(dilated,eroded)
cv2.imshow('edge image',edge)

#将图像二值化
retval,binary_result = cv2.threshold(edge,40,255,cv2.THRESH_BINARY)
cv2.imshow('binary edge image',binary_result)

#将二值图片取反
reverse = cv2.bitwise_not(binary_result)
cv2.imshow('reverse binary edge image',reverse)

cv2.waitKey(0)
cv2.destroyAllWindows()
