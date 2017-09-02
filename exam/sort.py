#coding:utf-8
"""
数组排序
"""
def bubble(a):
	for i in range(len(a)-1):
		k = 0
		for j in range(len(a)-i-1):
			if (a[j] > a[j+1]):
				k = a[j+1]
				a[j+1] = a[j]
				a[j] = k
	return a

print(bubble([9,8,7,6,5,4,3,2,1]))	


def fast_sort(a):
	low = []
	high = []
	median = []
	if len(a)<=1:
		return a
	else:
		baseline = a[0]
		for i in a:
			if i > baseline:
				high.append(i)
			elif i < baseline:
				low.append(i)
			else:
				median.append(i)
		low_arr = fast_sort(low)
		high_arr = fast_sort(high)
		return low_arr+median+high_arr

print(fast_sort([9,8,7,6,5,4,3,2,1]))	
