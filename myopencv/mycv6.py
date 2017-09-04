#coding:utf-8
"""
opencv绘制直方图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/ha1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.equalizeHist使图像均衡化
equal = cv2.equalizeHist(img)
cv2.imshow('hist equlization',np.hstack([img,equal]))
cv2.waitKey(0)
cv2.destroyAllWindows()
#[0]表示颜色通道,mask=None,灰度级数,横轴范围
hist = cv2.calcHist([img],[0],None,[256],[0,255])
#绘制均衡化后图片的直方图
hist1 = cv2.calcHist([equal],[0],None,[256],[0,255])
plt.subplot(1,2,1)
plt.title("hist picture of ha1")
plt.xlabel("bins")#X轴标签
plt.ylabel("num of bins")#Y轴标签
plt.plot(hist)
#设置x轴范围
plt.xlim([0,256])
plt.subplot(1,2,2)
plt.title("hist picture of ha1")
plt.xlabel("bins")#X轴标签
plt.ylabel("num of bins")#Y轴标签
plt.plot(hist1)
#设置x轴范围
plt.xlim([0,256])
plt.show()


"""
#彩色图直方图显示
img = cv2.imread('images/ha1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#创建一个窗口
cv2.namedWindow('ha1')
cv2.imshow('ha1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
channels = cv2.split(img)
colors = ('blue','green','red')
plt.figure()
plt.title('hist picture of ha1')
plt.xlabel('bins')
plt.ylabel('num of bins')

for (channel,color) in zip(channels,colors):
	hist = cv2.calcHist([channel],[0],None,[256],[0,255])
	plt.plot(hist,color=color)
	plt.xlim([0,256])
plt.show()
"""
