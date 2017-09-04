#coding:utf-8
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)

bhp_returns = np.diff(bhp) / bhp[ : -1] #diff()函数计算数组的差值，bhp[:-1]数组去掉最后一个值

vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unpack=True)

vale_returns = np.diff(vale) / vale[ : -1]

covariance = np.cov(bhp_returns, vale_returns) #np.cov()计算两个向量的协方差
print "Covariance", covariance

print "Covariance diagonal", covariance.diagonal() #diagonal()找出矩阵的对角元素
print "Covariance trace", covariance.trace() #trace()将矩阵对角元素求和

print covariance/ (bhp_returns.std() * vale_returns.std()) #bhp_returns.std()计算向量的标准差,var()方差,计算两个向量的相关系数

print "Correlation coefficient", np.corrcoef(bhp_returns, vale_returns) #np.corrcoef()计算两个向量的标准相关系数

difference = bhp - vale
avg = np.mean(difference)#np.mean()计算数组的平均值
dev = np.std(difference)#np.std()计算数组的标准差

print "Out of sync", np.abs(difference[-1] - avg) > 2 * dev #np.abs()求数的绝对值

t = np.arange(len(bhp_returns))
plot(t, bhp_returns, lw=1)
plot(t, vale_returns, lw=2)
show()
