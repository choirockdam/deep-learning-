#coding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
import datetime


#定义一个drawmatches函数
def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (201, 186, 131), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (28, 127, 135), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (137, 69, 148), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


beaver = cv2.imread('images/haha.PNG')
#彩色图
#plt.imshow(cv2.cvtColor(beaver,cv2.COLOR_BGR2RGB))
#灰度图
gray = cv2.cvtColor(beaver,cv2.COLOR_BGR2GRAY)
#SIFT特征检测器
#sift = cv2.xfeatures2d.SIFT_create()
sift=cv2.SIFT()
#sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(beaver,None)
#在图片中将特征点显示
beaver_sift = cv2.drawKeypoints(beaver,keypoints,None)
#找到关键点和描述符(128*kp)
kp,des = sift.compute(gray,keypoints)
plt.imshow(cv2.cvtColor(beaver_sift, cv2.COLOR_BGR2RGB))
plt.show()
#特征点匹配
#img1 = cv2.imread('images/box.png')
#img2 = cv2.imread('images/box_in_scene.png')
#img1 = cv2.imread('images/haha1.PNG')
#img2 = cv2.imread('images/haha.PNG')
img1 = cv2.imread('images/jianjian.jpg')
img2 = cv2.imread('images/jianjian1.jpg')
"""
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
"""
"""
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB(1000, 1.2)

(kp1,des1) = orb.detectAndCompute(gray1, None)

(kp2,des2) = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key=lambda val: val.distance)

out = drawMatches(gray1, kp1, gray2, kp2, matches[:30])

"""


"""
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)


des1 = des1 / np.repeat(np.sum(des1, axis = 1).reshape(des1.shape[0], 1), des1.shape[1], axis=1)
des2 = des2 / np.repeat(np.sum(des2, axis = 1).reshape(des2.shape[0], 1), des2.shape[1], axis=1)

# Calculate Hellinger distance for every feature pair
dist_mat = np.sqrt(np.abs(1.0 - np.dot(np.sqrt(des1), np.sqrt(des2).transpose())))

# Match with ratio test
min_arg = np.argsort(dist_mat, axis=1)
good_matches = []
for i in range(dist_mat.shape[0]):
    m, n = min_arg[i][0:2]
    if dist_mat[i][m] < dist_mat[i][n] * 0.75:
        dmatch = cv2.DMatch(i, m, 0, dist_mat[i][m]) # _queryIdx, _trainIdx, _imgIdx, _distance
        good_matches.append(dmatch)


out = drawMatches(gray1, kp1, gray2, kp2, good_matches[:30])

"""









