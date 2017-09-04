#coding:utf-8
import numpy as np
A = np.mat('1 2 3 4;5 6 7 8;9 8 7 6')
print(A)       #创建矩阵专用函数mat,每一行用分号隔开,每一列之间用空格隔开
B = A.T
print(B) #A.T将矩阵A转置
C = A.I #A.I求矩阵A的逆矩阵
print(C)
print('creation from array:',np.mat(np.arange(12).reshape(3,4))) #根据数组创建矩阵
print('***'*20)
D = np.eye(4) #创建一个4*4的单位矩阵
print(D)
E = 2*D
print(E)
print(np.bmat('D E;D E')) #使用字符串创建复合矩阵


