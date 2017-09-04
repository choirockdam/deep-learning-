#coding:utf-8
import numpy as np
import sys
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

#运行程序的方法：python sma.py 5，参数5以下值比较平滑
N = int(sys.argv[1]) #接受系统输入第二个参数

weights = np.ones(N) / N #设置权重，权重相同,权重个数为N
print "Weights", weights

c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)
sma = np.convolve(weights, c)[N-1:-N+1] #和t以及c[N-1:]的长度相同
print(sma.shape)
#print(len(c))
t = np.arange(N - 1, len(c))
#print(t)
#print(c[N-1:])
plot(t, c[N-1:], lw=1.0)
plot(t, sma, lw=2.0)
show()
