#coding:utf-8
"""
sklearn 0.18
python 3
KNN参数调优
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist/',one_hot=False)
#小数据量加快gridsearch的速度
x = mnist.train.images[0:1000,:]
y = mnist.train.labels[0:1000]
train_data,validation_data,train_labels,validation_labels = train_test_split(x,y,test_size=0.1)

#开始调优使用GridSearchCV找到,最优参数
knn = KNeighborsClassifier()
#设置k的范围
k_range = list(range(1,10))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
gridKNN = GridSearchCV(knn,param_gridknn,cv=10,scoring='accuracy',verbose=1)
gridKNN.fit(train_data,train_labels)
print('best score is:',str(gridKNN.best_score_))
print('best params are:',str(gridKNN.best_params_))