#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N=10000
lognormal_values = np.random.lognormal(size=N) #size=N,表示生成N个对数正态分布数
count, bins, _ = plt.hist(lognormal_values, np.sqrt(N), normed=True, lw=1) #count表示bin的个数,normed=True,是可以拟合的关键 
sigma = 1
mu = 0
x = np.linspace(min(bins), max(bins), len(bins)) 
pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)) #对数正态分布公式
plt.plot(x, pdf,lw=3)
plt.show()
