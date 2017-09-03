#coding:utf-8
"""
输入一个整数n，求从1到n这n个正数中，1出现的次数。例如：输入12，出现一的数字有1，10，11，12共有5个1，则输出5.
"""
def count_number_one(n):
	count = 0
	for i in range(n+1):
		while i:
			if i % 10 == 1:
				count += 1
			i = i/10
	return count
print(count_number_one(12))

