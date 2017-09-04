#coding:utf-8
import numpy as np
import sys
from matplotlib.pyplot import plot
from matplotlib.pyplot import show


bhp=np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)

vale=np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unpack=True)

t = np.arange(len(bhp))
poly = np.polyfit(t, bhp - vale, int(sys.argv[1])) #np.polyfit()拟合函数,int(sys.argv[1])参数表示拟合为几次多项式,越多拟合效果越好
print "Polynomial fit", poly #poly为拟合后返回的系数

print "Next value", np.polyval(poly, t[-1] + 1) #np.polyval(poly,t[-1]+1)根据多项式对象预测下一个值

print "Roots", np.roots(poly) #np.roots(poly)根据多项式对象求根

der = np.polyder(poly) #np.polyder(poly)根据多项式对象求倒数，返回的是倒数系数组成的多项式对象
print "Derivative", der

print "Extremas", np.roots(der) #np.roots(der)根据多项式求根
vals = np.polyval(poly, t) #np.polyval(poly,t)根据多项式对象求值
print np.argmax(vals) #np.argmax()返回最大值所对应的索引
print np.argmin(vals) #np.argmin()返回最小值所对应的索引

plot(t, bhp - vale)
plot(t, vals) #画出拟合的曲线
aa,bb = np.loadtxt('BHP.csv',delimiter=',',usecols=(6,7),unpack=True)
change = np.diff(aa)#np.diff()计算数组之间的差值
signs = np.sign(change) #np.sign()函数返回符号,数组值大于0返回1,小于0,返回-1
print(signs)
show()
