#coding:utf-8
"""
sklearn 0.18
python 3

"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import input_data
import numpy as np
import pickle

mnist = input_data.read_data_sets('mnist/',one_hot=False)
x = mnist.train.images
y = mnist.train.labels
#train_test_split设置一部分数据作为验证集
train_data,validation_data,train_labels,validation_labels = train_test_split(x,y,test_size=0.1)
#clf相当于一个采用KNN算法进行分类 m_neighbors=4表示k=4的分类器
clf = KNeighborsClassifier(n_neighbors=4,algorithm='auto',weights='distance')
clf.fit(train_data,train_labels)
predictions=[]
for i in range(1000):
    if i % 100 == 0:
        print('= = = = = = > > > > > >','epoch :',int(i/100))
    #满足输入规范要求clf.predict([])
    output = clf.predict([mnist.test.images[i]])
    predictions.append(output)
#混淆矩阵
print (confusion_matrix(mnist.test.labels[0:1000],predictions))
#f1-score,precision,recall
print (classification_report(mnist.test.labels[0:1000],np.array(predictions)))
#计算准确度
print ('test accuracy is :',accuracy_score(mnist.test.labels[0:1000],predictions))
#将训练好的分类器保存
with open('mnist_knn.pickle','wb') as f:
    pickle.dump(clf,f)