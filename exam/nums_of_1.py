#coding:utf-8
"""
输出数字的二进制中的1的个数; 判断一个数是否为2的幂
"""
def count_1(n):
	counter = 0
	while n:
		n = n&(n-1)
		counter +=1
	return counter

def power_2(n):
	if n&(n-1) == 0:
		return True
	else:
		return False

print(count_1(15))
print(power_2(8))
