#coding:utf-8
"""
将一个字符串转换成数字 
"""
def string_to_num(s):
	result = 0
	for i in range(len(s)):
		result += int(s[i])*pow(10,len(s)-i-1)
	return result

print(string_to_num('12345'))
