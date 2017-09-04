#coding:utf-8
import numpy as np
import sys
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

x = np.arange(5)
print "Exp", np.exp(x) #np.exp(x)可以计算数组x中每个元素的指数值
print "Linspace", np.linspace(-1, 0, 5)

N = int(sys.argv[1])


weights = np.exp(np.linspace(-1., 0., N))#np.exp()对数组中的每个值求指数
weights /= weights.sum() #权重的个数为N,权重不同 weights.sum()将数组中的所有值求和
print "Weights", weights

c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)
ema = np.convolve(weights, c)[N-1:-N+1] #卷积后的个数为[N-1:]
t = np.arange(N - 1, len(c))
plot(t, c[N-1:], lw=1.0)
plot(t, ema, lw=2.0)
show()
