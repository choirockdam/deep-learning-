#coding:utf-8
"""
从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2-10为数字本身，A为1，J为11，Q为12，K为13，而大小王可以看成任意数字。
"""
def is_shunzi(a):
	if len(a) <= 4:
		return False
	if list(set(a)) == [0]:
		return True
	hash_dict = dict()
	for i in a:
		if i == 0:
			continue
		if i in hash_dict.keys():
			return False
		else:
			hash_dict[i] = 100
	result = True if (max(hash_dict.keys())-min(hash_dict.keys())) <= 4 else False
	return result

a = input()
data = [int(x) for x in a.split()] 
print(is_shunzi(data))
