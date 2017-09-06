#coding:utf-8
"""
python 3 
sklearn 0.18
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import input_data
import numpy as np
import pickle

mnist = input_data.read_data_sets('mnist/',one_hot=False)
x = mnist.train.images
y = mnist.train.labels
#采用交叉验证
train_data,validation_data,train_labels,validation_labels = train_test_split(x,y,test_size=0.2)
#训练一个DecisionTree分类器
clf = DecisionTreeClassifier(random_state=0,splitter='best',criterion='entropy')
clf.fit(train_data,train_labels)
predictions=[]
for i in range(1000):
    if i % 100 ==0:
        print('= = = = = = > > > > > >','epoch:',int(i/100))
    #将预测结果存入predictions
    output = clf.predict([mnist.test.images[i]])
    predictions.append(output)
print(confusion_matrix(mnist.test.labels[0:1000],predictions))
print(classification_report(mnist.test.labels[0:1000],np.array(predictions)))
print('test accuracy is:',accuracy_score(mnist.test.labels[0:1000],predictions))
with open('id3.pickle','wb') as f:
    pickle.dump(clf,f)