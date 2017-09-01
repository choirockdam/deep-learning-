#coding:utf-8
"""
用递归实现一个字符串的全排列
"""
def arrangement(s):
	arr_s = list(s)
	if len(arr_s) == 0:
		return [[]]
	else:
		result = []
		for i in arr_s:
			temp = arr_s[:]
			temp.remove(i)
			for j in arrangement(temp):
				result.append([i]+j)
	return result

print([''.join(x) for x in arrangement('abc')])
