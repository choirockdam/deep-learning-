#coding:utf-8
"""
将两个排序的数组merge成新的排好序的数组
"""
def resort(a,b):
	result = []
	n1 = len(a)
	n2 = len(b)
	n = n1+n2
	i = 0
	j = 0
	for index in range(n):
		if (i < n1 and j < n2):
			if a[i] > b[j]:
				result.append(b[j])
				j = j+1
			else:
				result.append(a[i])
				i = i+1
		elif i >= n1:
			result.append(b[j])
			i = j+1
		elif j >= n2:
			result.append(a[i])
			i = i+1
	return result
print(resort([1,2,3,4],[2,3,4,5]))

	
