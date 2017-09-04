#coding:utf-8
import numpy as np
#import datetime
from datetime import datetime

def datestr2num(s):
   return datetime.strptime(s, "%d-%m-%Y").toordinal() #toordinal 返回数字形式的字符串
   #datetime.datetime.strptime(s,'%d-%m-%Y')将字符串按如下格式整理
   #return datetime.datetime.strptime(s,'%d-%m-%Y').date.weekday() 如果是星期一则返回0	
dates,closes=np.loadtxt('AAPL.csv', delimiter=',', usecols=(1, 6), converters={1:datestr2num}, unpack=True)
indices = np.lexsort((dates, closes)) #np.lexsort(a,b)根据b从小到大排序,如果相同根据a排序

print "Indices", indices
print ["%s %s" % (datetime.fromordinal(int(dates[i])),  closes[i]) for i in indices] #datetime.fromordinal()转为标准时间格式
