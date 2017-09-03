#coding:utf-8
"""
最大连续子序列和
"""
def get_continuous(a):
	n = len(a)
	max_value = 0
	result = []
	for i in range(n):
		temp = 0
		for j in range(i,n):
			temp += a[j]
			if temp > max_value:
				max_value = temp
				result = a[i:j+1]
	return result

print(get_continuous([31,-41,59,26,-53,58,97,-93,-23,84]))


