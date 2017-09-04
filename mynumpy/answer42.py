#coding:utf-8
import numpy as np

def ultimate_answer(a):
   result = np.zeros_like(a) #np.zeros_like(a) 创建一个和a一样形状的0数组
   result.flat = 42 #result.flat扁平迭代器为数组中每一个元素赋值

   return result

ufunc = np.frompyfunc(ultimate_answer, 1, 1)  #np.frompyfunc将ultimate_answer创建为通用函数,输入参数为1,输出参数为1
print "The answer", ufunc(np.arange(4))

print "The answer", ufunc(np.arange(4).reshape(2, 2))

print('***'*20)
a = np.arange(9)
print('reduce:',np.add.reduce(a)) #np.add.reduce()对数组的计算结果求和
print('accumulate:',np.add.accumulate(a)) #np.add.accumulate()对输入的数组进行累加求和
print('Reducate',np.add.reduceat(a,[0,5,2,7])) #第一次累加[0:5],第二次计算索引为5的值,第三次计算[2:7],第三次计算[7:]

print('+++++'*15)
a = np.array([2,6,5]) #np.array()创建一个数组
b = np.array([1,2,3])
print(np.divide(a,b)) #np.divide()整除
print(np.divide(b,a))
print(np.true_divide(a,b)) #np.true_divide()非整除,相当于from __future__ import division
print(np.true_divide(b,a)) 
print('==='*20)
a = np.arange(-4,4)
print(np.remainder(a,2))  #np.remainder(a,2)对数组a求余
print(np.mod(a,2))
print(a%2)


