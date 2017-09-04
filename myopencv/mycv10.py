#coding:utf-8
"""
opencv滤波器
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/white.jpg',0)
#cv2.blur低通滤波器的目标是降低图像的变化率,低通滤波器大小5*5
blur = cv2.blur(img,(5,5))
#cv2.boxFilter滤波器,-1表示输出图像深度与输入图像相同
boxfilter = cv2.boxFilter(img,-1,(5,5))
#cv2.GaussianBlur高斯滤波器,滤波器模板大小为奇数,1.5表示横向纵向滤波系数
gaussianblur = cv2.GaussianBlur(img,(5,5),1.2)

def salt(img,n):
	for k in range(n):
		#x坐标
		i = int(np.random.rand()*img.shape[0])
		#y坐标
		j = int(np.random.rand()*img.shape[1])
		if img.ndim == 2:
			img[i,j] = 255
		elif img.ndim == 3:
			img[i,j,0] = 255
			img[i,j,1] = 255
			img[i,j,2] = 255
	return img
cv2.imshow('original img',img)
cv2.imshow('blur img',blur)
cv2.imshow('boxfilter img',boxfilter)
cv2.imshow('gaussianblur img',gaussianblur)

salt = salt(img,1000)
#cv2.medianBlur中值滤波器消除噪点
medianblur = cv2.medianBlur(img,5)
cv2.imshow('salt img',salt)
cv2.imshow('medianblur img',medianblur)
cv2.waitKey(0)
cv2.destroyAllWindows()

