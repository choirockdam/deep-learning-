#coding:utf-8
import numpy as np
import sys
import matplotlib.pyplot as plt
N = int(sys.argv[1]) 

weights = np.hanning(N) #np.hanning(N),计算权重生成长度为N的窗口
print "Weights", weights

bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
bhp_returns = np.diff(bhp) / bhp[ : -1] #np.diff()计算数组差值
smooth_bhp = np.convolve(weights/weights.sum(), bhp_returns)[N-1:-N+1] #np.convolve()[n-1:-N+1]计算数组卷积长度为[N-1:]

vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unpack=True)
vale_returns = np.diff(vale) / vale[ : -1]
smooth_vale = np.convolve(weights/weights.sum(), vale_returns)[N-1:-N+1]

K = int(sys.argv[1]) #拟合几次多项式
t = np.arange(N - 1, len(bhp_returns))
poly_bhp = np.polyfit(t, smooth_bhp, K) #np.plotfit()返回拟合多项式的系数
poly_vale = np.polyfit(t, smooth_vale, K) 

poly_sub = np.polysub(poly_bhp, poly_vale) #np.plotsub()计算两个多项式的差构成的多项式
poly_sub_val = np.polyval(poly_sub,t)
poly_der = np.polyder(poly_bhp)
poly_der_val = np.polyval(poly_der,t)
xpoints = np.roots(poly_sub) #np.roots()计算多项式的根
print "Intersection points", xpoints

reals = np.isreal(xpoints) #np.isreal()判断多项式的根是否为实数,并返回布尔值
print "Real number?", reals

xpoints = np.select([reals], [xpoints]) #np.select([reals],[xpoints])将xpoints数组中实数值返回,虚数值返回0
xpoints = xpoints.real #返回数组中标签为实数的值
print "Real intersection points", xpoints

print "Sans 0s", np.trim_zeros(xpoints)#np.trim_zeros()去除一维数组中开头和末尾的0元素
plt.figure()
plt.subplot(3,1,1)
plt.plot(t, bhp_returns[N-1:], lw=1.0,label='bhp_return normal data')
plt.plot(t, smooth_bhp, lw=2.0,label='hanning smoothng')
plt.grid(True)
plt.title('bhp_returns polyfit and smoothing')
plt.legend(loc='best')
plt.ylim(-0.05,0.05,2)
plt.yticks([-0.04,-0.02,0.00,0.02,0.04],['100M','200M','300M','400M','500M'])
plt.subplot(3,1,2)
plt.plot(t, vale_returns[N-1:], lw=1.0,label='vale_return nornal data')
plt.plot(t, smooth_vale, lw=2.0,label='hanning smoothing')
plt.grid(True)
plt.title('vale_returns polyfit and smoothing')
plt.legend(loc='best')
plt.ylim(-0.05,0.05,2)
plt.yticks([-0.04,-0.02,0.00,0.02,0.04],['100M','200M','300M','400M','500M'])
plt.subplot(3,1,3)
plt.plot(t,poly_sub_val,label='polysub')
plt.plot(t,poly_der_val,label='polyval')
plt.title('polysub and polyder')
plt.grid(True)
plt.legend(loc='best')
plt.show()
