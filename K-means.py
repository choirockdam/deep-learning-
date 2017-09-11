#coding:utf-8
"""
python 3
sklearn 0.18
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
import input_data
import numpy as np
import pickle


iris = load_iris()
x = iris.data
y = iris.target
#mnist = input_data.read_data_sets('mnist/',one_hot=False)
#x = mnist.train.images
#y = mnist.train.labels
#训练一个Kmeans分类器
clf = KMeans(n_clusters=3)
clf.fit(x)
predictions = clf.predict(x)
"""
predictions = []
for i in range(1000):
    if i % 100 ==0:
        print('= = = = = = > > > > > >','epoch:',int(i/100))
    output = clf.predict([mnist.test.images[i]])
    predictions.append(output)
"""
print(confusion_matrix(y,predictions))
print(classification_report(y,np.array(predictions)))
print('test accuracy is:',accuracy_score(y,predictions))