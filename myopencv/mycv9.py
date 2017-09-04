#coding:utf-8
"""
opencv形态学检测拐角
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取灰度图
img = cv2.imread('images/white.jpg',0)
cv2.imshow('original img',img)

#定义十字型
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#定义菱形
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel2[0, 0] = 0  
kernel2[0, 1] = 0  
kernel2[1, 0] = 0  
kernel2[4, 4] = 0  
kernel2[4, 3] = 0  
kernel2[3, 4] = 0  
kernel2[4, 0] = 0  
kernel2[4, 1] = 0  
kernel2[3, 0] = 0  
kernel2[0, 3] = 0  
kernel2[0, 4] = 0  
kernel2[1, 4] = 0 
#定义方形
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


kernel4 = np.uint8(np.zeros((5,5)))
kernel4[0,0] = 1
kernel4[0,4] = 1
kernel4[1,1] = 1
kernel4[1,3] = 1
kernel4[2,2] = 1
kernel4[3,1] = 1
kernel4[3,3] = 1
kernel4[4,0] = 1
kernel4[4,4] = 1


#使用十字形状膨胀图片
dilated1 = cv2.dilate(img,kernel1)
#使用菱形腐蚀图像
eroded1 = cv2.erode(dilated1,kernel2)

#使用十字型膨胀图片
dilated2 = cv2.dilate(img,kernel4)
#使用方型腐蚀图像
eroded2 = cv2.erode(dilated2,kernel3)
#获得拐角
result = cv2.absdiff(eroded1,eroded2)
#将拐角二值化
retval,binary_result = cv2.threshold(result,40,255,cv2.THRESH_BINARY)
#print(binary_result.size)
#print(binary_result.shape)
#在图上将拐角标出
for j in range(binary_result.size):
	y = j / binary_result.shape[0]
	x = j % binary_result.shape[0]

	if binary_result[x,y] == 255:
		cv2.circle(img,(y,x),5,(255,0,0))
cv2.imshow('result img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()











