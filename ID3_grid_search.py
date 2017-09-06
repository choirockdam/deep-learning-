#coding:utf-8
"""
python 3
scikit-learn 0.18
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist/',one_hot=False)
x = mnist.train.images
y = mnist.train.labels

train_data,validation_data,train_labels,validation_labels = train_test_split(x,y,test_size=0.2)
#使用GridSearchCV找到最优参数
dtree = DecisionTreeClassifier(random_state=0)
#gini ,表示决策树非叶节点划分依据是根 据 gini 指数表示划分的纯度。
#entropy ,用信息增益来衡量 划分的优劣
criterion_options = ['gini','entropy']
splitter_options = ['best','random']
param_griddtree = dict(criterion=criterion_options,splitter=splitter_options)
griddtree = GridSearchCV(dtree,param_griddtree,cv=10,scoring='accuracy',verbose=1)
griddtree.fit(train_data,train_labels)
print('best score is:',str(griddtree.best_score_))
print('best params are :',str(griddtree.best_params_))