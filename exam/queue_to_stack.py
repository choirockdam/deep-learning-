#coding:utf-8
"""
两个队列实现栈
"""
class Stack(object):
	def __init__(self):
		self.q1 = []
		self.q2 = []
	
	def in_stack(self,data):
		while self.q2:
			self.q1.append(self.q2.pop(0))
		self.q1.append(data)
		print(data,'------in stack')
	
	def out_stack(self):
		while self.q1:
			self.q2.append(self.q1.pop(0))
		try:
			print(self.q2.pop(),'------out stack')
		except:
			print('empty')

if __name__ == '__main__':
	stack = Stack()
	stack.in_stack(0)
	stack.in_stack(1)
	stack.in_stack(2)
	stack.out_stack()
	stack.out_stack()
	stack.in_stack(3)
	stack.out_stack()
	stack.out_stack()
	















