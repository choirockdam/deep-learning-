#coding:utf-8
"""
输入一个正整数数组，将它们连接起来排成一个数，输出能排出的所有数字中最小的一个。例如输入数组{32, 321}，则输出这两个能排成的最小数字32132
"""
def arrangement(a):
	if len(a) == 0:
		return [[]]
	else:	
		result = []
		for i in a:
			temp = a[:]
			temp.remove(i)
			for j in arrangement(temp):
				result.append([i]+j)
		return result

def get_min_number(a):
	s = ''.join(a)
	s_arr = list(s)
	result = arrangement(s_arr)
	k = int(''.join(result[0]))
	for i in result[1:]:
		if int(''.join(i))< k:
			k = int(''.join(i))	
	return k
print(get_min_number(['321','32','3']))

	
