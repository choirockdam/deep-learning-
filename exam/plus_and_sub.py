#coding:utf-8
"""
没有加法运算符的加法实现， 没有减法运算符实现减法运算
"""
def plus_nums(a,b):
	if a == 0:
		return b
	elif b == 0 :
		return a
	else:
		xor_ab = a^b
		and_right = (a&b) << 1
		return plus_nums(xor_ab,and_right)

def sub_nums(a,b):
	if a == b:
		return 0
	else:
		inverse_b = plus_nums(~b,1)
		return plus_nums(a,inverse_b) 

print(plus_nums(4,6)) 
print(sub_nums(4,6))
