#coding:utf-8
"""
递归求解1+2+......n
"""
def add(n):
	if n == 1:
		return n
	else:
		return add(n-1)+n
print(add(100))
