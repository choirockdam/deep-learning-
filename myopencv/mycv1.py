#coding:utf-8
"""
基本操作
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

#导入图片
img = cv2.imread('images/messi.jpg',cv2.IMREAD_COLOR)
if img is None:
	print('error')
else:
	print('ok')
print(type(img))
print(img.shape)
print(img.dtype)
#显示图片
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.show()

img2 = np.ones((512,512,3),np.uint8)*254
#img3 = np.zeros((512,512,3),np.uint8)
#plt.imshow(img2)
#在图上画线条,起始位置,终止位置,颜色,粗细
cv2.line(img2,(0,0),(511,511),(128,314,50),5)
cv2.rectangle(img2,(100,200),(400,400),(17,25,133),3)
cv2.circle(img2,(200,200),23,(230,120,25),3)
cv2.ellipse(img2,(256,256),(100,50),-45,0,180,(25,129,120),-1)
pts = np.array([[10,20],[150,200],[300,150],[200,50]],np.int32)
pts = pts.reshape((-1,1,2))
#画任意形状的线isClosed=True
cv2.polylines(img2,[pts],True,(0,255,255),3)
#给图片添加文字LINE_AA change to CV_AA
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img2,'SZJ-JOJO',(40,500),font,3,(254,34,12),cv2.LINE_AA)
#修改图片颜色
for i in range(5):
	for j in range(5):
		img[170+i,220+j] = (22,111,0)
#font1 = cv2.FONT_HERSHEY_PLAIN
#cv2.putText(img,'haha',(171,221),font1,1,(0,0,0),cv2.CV_AA)
#改变图中物体位置
ball = img[280:340,330:390]
img[273:333,100:160] = ball
img[270:330,5:65] = ball
img[270:330,160:220] = ball

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.imshow(img2)
plt.show()




























