#coding:utf-8
"""
1*2*3*......n
"""
def factorial(n):
	if n == 1:
		return n
	else:
		return factorial(n-1)*n

print(factorial(5))
