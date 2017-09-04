#coding:utf-8
import numpy as np
A = np.mat('1 2 3;4 5 6;7 8 9') #np.mat('')构造矩阵
print(A)
inverse = np.linalg.inv(A) #np.linalg.inv()求矩阵A的逆矩阵
print(inverse)
print('---'*20)
B = np.mat('1 -2 1;0 2 -8;-4 5 9')
print(B)
C = np.array([0,8,-9])
print(C)
x = np.linalg.solve(B,C) #np.linalg.solve(B,C)求解Bx=C的解
print(x)
print(np.dot(B,x)) #求解Ax的点乘积
print('^^^'*20)
a = np.mat('-3 -2;1 0')
print(a)
b = np.linalg.eigvals(a)  #np.linalg.eigvals()求解矩阵a的特征值
eigvalues,eigvectors = np.linalg.eig(a) #np.linalg.eig()求解矩阵a的特征值和特征向量
print(b)
print(eigvalues)
print(eigvectors)
dd = np.mat('3 4;5 6')
print(dd)
print(np.linalg.det(dd))  #np.linalg.det()求矩阵的行列式
