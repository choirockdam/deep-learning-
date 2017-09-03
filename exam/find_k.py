#coding:utf-8
"""
寻找第K大数
"""
def fast_sort(a):
	low = []
	high = []
	median = []
	if len(a) < 1:
		return a
	else:
		baseline = a[0]
		for i in a:
			if i < baseline:
				low.append(i)
			elif i > baseline:
				high.append(i)
			else:
				median.append(i)
		low_arr = fast_sort(low)
		high_arr = fast_sort(high)
		return low_arr+median+high_arr

def find_k_number(a,k):
	resort_arr = fast_sort(a)
	return resort_arr[-k]

a = input()
data = [int(x) for x in a.split()]
k = data.pop()
print(find_k_number(data,k))


