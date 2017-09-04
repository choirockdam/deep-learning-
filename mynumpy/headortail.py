#coding:utf-8
import numpy as np
#from matplotlib.pyplot import plot, show 
import matplotlib.pyplot as plt

cash = np.zeros(10000) #创建10000个0的数组
cash[0] = 50
outcome = np.random.binomial(9, 0.5, size=len(cash)) #np.random.binomial(n,p,size)size表示生成的数,选出9的概率是0.5,其他值概率是0.5

for i in range(1, len(cash)):

   if outcome[i] < 5:
      cash[i] = cash[i - 1] - 0.3
   elif outcome[i] < 10:
      cash[i] = cash[i - 1] + 0.3
   else:
      raise AssertionError("Unexpected outcome " + outcome)

print outcome.min(), outcome.max()  #输出最大值,最小值
t = np.arange(len(cash))
plt.plot(t, cash,color='m',label='plt.fill_between')
plt.grid(True)
plt.title('fill_between')
plt.fill_between(t,cash,where = t>np.median(t),facecolor='red',alpha=0.6)
plt.fill_between(t,cash,where = t<=np.median(t),facecolor='green',alpha=0.6)
plt.legend(loc='best')
plt.ylim(0,100)
plt.yticks([0,20,40,60,80,100],['0M','20M','40M','60M','80M','100M'])
plt.show()
