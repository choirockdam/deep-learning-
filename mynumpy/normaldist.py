#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N=10000

normal_values = np.random.normal(size=N) #size=N表示返回N个高斯随机数,第一个参数为均值,第二个参数为标准差，缺省表示均值为0标准差为1
#dummy, bins, dummy = plt.hist(normal_values, np.sqrt(N), normed=True, lw=1) 
#plt.hist()输入第一个参数表示数据点,第二个参数表示分为多少个bin,normed=True表示标准化,以概率计算
#plt.hist()输出三个参数,第一个参数表示在每个bin中的数据点个数,第二个参数表示bin的中间值,第三个参数预留值
count,bins,_ = plt.hist(normal_values,np.sqrt(N),normed=True,label='hist') #count表示计算某一bin出现的次数,100个bins
plt.title('normal hist')
#print('111',dummy,'222',bins,'333',dummy)
sigma = 1 #标准差
mu = 0 #均值
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),lw=2,color='r') #标准正态分布
plt.xlim(-5,5,8)
plt.grid(True)
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5],['0M','10M','30M','30M','40M','50M'])
plt.legend(loc='best')
plt.show()
