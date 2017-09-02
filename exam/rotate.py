#coding:utf-8
"""
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个排序的数组的一个旋转（递增或递减的），输出旋转数组的最小元素。例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
"""
def rotate_arr(a):
	left = 0
	right = len(a)-1
	mid = (left+right)/2
	if a[right] - a[left] !=1 and a[right] - a[left] != -1:
		return a[left]
	if a[left] == a[mid] or a[left] == a[right] or a[right] == a[mid]:
		return min(a)
	while left < right:
		middle = (left+right)/2
		if a[left] < a[middle]:
			left = middle
		elif a[left] > a[middle]:
			right = middle
		elif a[left] == a[middle] and right-left == 1:
			return a[right] 		
	
#print(rotate_arr([0,1,1,1,0,0]))
#print(rotate_arr([3,4,5,6,7,1,2]))
print(rotate_arr([1,2,3,4,5,6,7]))
				
			
