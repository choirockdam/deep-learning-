#coding:utf-8
"""
求菲波那切数列1,1,2,3,5,8,13,21......
"""
def fibo(n):
	if n == 1:
		return 1
	elif n == 2:
		return 1
	else:
		return fibo(n-1)+fibo(n-2)

print(fibo(8))
