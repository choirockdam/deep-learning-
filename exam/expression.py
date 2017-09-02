#coding:utf-8
"""
输入为一行字符串，即一个表达式。其中运算符只有-,+,*。参与计算的数字只有0~9
"""

def str_isalnum(s):
	return s.isalnum()

def compute_expression(s):
	arr = list(s)
	num = []
	operator = []
	for i in range(len(arr)/2):
		
		num.append(arr[2*i])
		
		operator.append(arr[2*i+1])
	num.append(arr[-1])
	k = num[0]
	for i in range(len(operator)):
		k = eval(str(k)+str(operator[i])+str(num[i+1]))
	return k

print(compute_expression('3+5*7'))

