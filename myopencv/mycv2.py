#coding:utf-8
"""
边缘检测
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

#cv2.IMREAD_GRAYSCALE以灰度形式导入图片
#img = cv2.imread('images/wheel.png',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('images/messi.jpg',cv2.IMREAD_GRAYSCALE)
print(img.dtype)
print(img.shape)
#dst = cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1.0)
#图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
ddepth = cv2.CV_64F
#求导的阶数
dx = 1
dy = 0
sobel_img = cv2.Sobel(img,ddepth,dx,dy)

"""
sobelx = cv2.Sobel(img,ddepth,1,0)
sobely = cv2.Sobel(img,ddepth,0,1)
#alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值
sobel = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx,2.0),1.0,cv2.pow(sobely,2.0),1.0,0.0))
plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.title('input image:')
plt.axis('off')
plt.imshow(img,cmap='gray')
plt.subplot(2,2,2)
plt.title('sobelx image:')
plt.axis('off')
#sobelx,sobely可能为负数值,cv2.absdiff() 差的绝对值
plt.imshow(cv2.absdiff(sobelx,0.0),cmap='gray')
plt.subplot(2,2,3)
plt.title('sobely image:')
plt.axis('off')
plt.imshow(cv2.absdiff(sobely,0.0),cmap='gray')
plt.subplot(2,2,4)
plt.title('sobel image:')
plt.axis('off')
plt.imshow(sobel,cmap='gray')
#plt.imshow(sobel_img,cmap='gray')
#plt.imshow(img,cmap='gray')
"""


threshold1 = 100
threshold2 = 150
canny = cv2.Canny(img,threshold1,threshold2)
plt.imshow(canny,cmap='gray')

plt.show()






























