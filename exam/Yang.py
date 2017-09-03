#coding:utf-8
"""
在一个 m 行 n 列二维数组中, 每一行都按照从左到右递增的顺序排序, 每一列都按照从上到下递增的顺序排序.请完成一个函数, 输入这样的一个二维数组和一个整数, 判断数组中是否含有该整数
"""
def get_n(a,n):
	rows = len(a)
	cols = len(a[0])
	for i in range(rows):
		for j in range(cols):
			if a[rows-i-1][j] == n:
				return True
			elif a[rows-i-1][j] > n:
				continue
			elif a[rows-i-1][j] < n:
				if j == cols-1:
					return False
				elif a[rows-i-1][j+1] == n:
					return True
	return False
	

print(get_n([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],1))

			 
