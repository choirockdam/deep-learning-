#coding:utf-8
"""
两个栈实现队列
"""
class Queue(object):
	def __init__(self):
		self.s1 = []
		self.s2 = []
	
	def in_queue(self,data):
		if self.s2 == []:
			self.s2.append(data)
			print(data,'------in queue')
		else:
			while self.s2:
				self.s1.append(self.s2.pop())
			self.s1.append(data)
			print(data,'------in queue')
	def out_queue(self):
		while self.s1:
			self.s2.append(self.s1.pop())
		try:
			print(self.s2.pop(),'------out queue')
		except:
			print('empty')


if __name__ == '__main__':
	queue = Queue()
	queue.in_queue(0)
	queue.in_queue(1)
	queue.in_queue(2)
	queue.in_queue(3)
	queue.in_queue(4)
	queue.out_queue()
	queue.out_queue()
	queue.in_queue(5)
	queue.out_queue()










