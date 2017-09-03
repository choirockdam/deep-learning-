#coding:utf-8
"""
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位予数组的后半部分。
"""
def adjust_arr(a):
	left = 0
	right = len(a)-1
	k = 0
	while left < right:
		if (a[left] % 2 == 0 and a[right] % 2 !=0):
			k = a[right]
			a[right] = a[left]
			a[left] = k 
		left += 1
		right -= 1
	return a

#print(adjust_arr([2,4,6,8,10,1,3,5,7,9]))
print(adjust_arr([0,1,2,3,4,5,6,7,8,9]))
					
