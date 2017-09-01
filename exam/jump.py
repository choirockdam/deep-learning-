#coding:utf-8
def jump1(n):
	if n == 1:
		return 1
	elif n == 2:
		return 2
	else:
		return jump1(n-1)+jump1(n-2)

def jump2(n):
	if n == 1:
		return 1
	elif n == 2:
		return 2
	else:
		return 2*jump2(n-1)

print(jump1(5))
print(jump2(5))
