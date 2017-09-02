#coding:utf-8
"""
约瑟夫环问题的原来描述为，设有编号为1，2，……，n的n(n>0)个人围成一个圈，从第1个人开始报数，报到m时停止报数，报m的人出圈，再从他的下一个人起重新报数，报到m时停止报数，报m的出圈，……，如此下去，直到所有人全部出圈为止。当任意给定n和m后，设计算法求n个人出圈的次序
"""
def joseph(a,m):
	index = 0
	while len(a) != 1:
		index = (index+m-1)%len(a)
		print(a[index],'------out')
		a.remove(a[index])
	
	print(a,'------winner')
if __name__ == '__main__':
	a = [1,2,3,4,5,6]
	joseph(a,2)

