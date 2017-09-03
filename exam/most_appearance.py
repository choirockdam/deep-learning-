#coding:utf-8
"""
找出数组中出现次数最多的那个元素的值。如[1,2,3,4,5,6,2,2,2,2,27,8]结果为2
"""
def get_most_appearance(a):
	number_dict = dict()
	for i in range(max(a)+1):
		number_dict[i] = 0
	for i in a:
		number_dict[i] += 1
	temp = 0
	target_number = 0
	for key,value in number_dict.items():
		if temp < value:
			temp = value
			target_number = key
	return ['%s'%target_number,':::',temp]		

print(get_most_appearance([1,2,3,1,1,1,1,1,4,13,3,5]))

