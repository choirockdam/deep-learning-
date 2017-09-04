#coding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

#级联分类器路径
cascpath = 'detect/haarcascade_frontalface_alt.xml'
facecascade = cv2.CascadeClassifier(cascpath)

img = cv2.imread('images/ioi2.jpg')
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.show()
def detect_faces_show(path):
	img = cv2.imread(path)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	faces = facecascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
	print('fond faces:',len(faces))
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(114,120,2),4)
	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	plt.show()


detect_faces_show('images/telangpu.jpg')
































