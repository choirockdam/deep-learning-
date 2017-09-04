#coding:utf-8
"""
opencv 基本操作
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


#cv2.imread()
img = cv2.imread('images/ha1.jpg')
#创建一个窗口
cv2.namedWindow('newwindow')
cv2.imshow('newwindow',img)
cv2.waitKey(0)
#释放窗口
cv2.destroyAllWindows()

#创建一个图像
emptyimage = np.ones(img.shape,np.uint8)*255
#复制图像
emptyimage1 = img.copy()
#复制图片2
emptyimage2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#保存图片
#cv2.imwrite('/home/tuoxin/emptyimage2.jpg',emptyimage2)
#并压缩图片
#cv2.imwrite('/home/tuoxin/emptyimage1-1.png',emptyimage1,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
#cv2.imwrite('/home/tuoxin/emptyimage1-2.png',emptyimage1,[int(cv2.IMWRITE_PNG_COMPRESSION),9])

#判断图片是彩色还是灰度图
for k in range(10000):
	i = np.random.rand()*emptyimage.shape[0]
	j = np.random.rand()*emptyimage.shape[1]
	if emptyimage.ndim == 2:
		emptyimage[i,j] = 0
	elif emptyimage.ndim == 3:
		emptyimage[i,j,0] = 0
		emptyimage[i,j,1] = 0
		emptyimage[i,j,2] = 0
cv2.namedWindow('newwindow1')
cv2.imshow('newwindow1',emptyimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#将图片颜色通道分离
b,g,r = cv2.split(emptyimage1)
#print(b[0:50,0:50])
#print(g[0:50,0:50])
#print(r[0:50,0:50])

#将图片颜色通道合并
merged = cv2.merge([b,g,r])
merged_np = np.dstack([b,g,r])
print(merged.shape)
print(merged_np.shape)










































































































