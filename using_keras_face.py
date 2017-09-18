#coding:utf-8
"""
python 2.7
keras 2.04
"""
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix,classification_report
import input_faces
import datetime

start_time = datetime.datetime.now()
#设置随机种子
np.random.seed(1000)
(train_data,train_labels),(test_data,test_labels) = input_faces.read_data_sets('olivettifaces.gif')
#reshape
train_data = train_data.reshape(train_data.shape[0],1,57,47)
train_labels = np_utils.to_categorical(train_labels,nb_classes=40)
test_data = test_data.reshape(test_data.shape[0],1,57,47)
test_labels = np_utils.to_categorical(test_labels,nb_classes=40)

#构建模型
model = Sequential()
#(None,1,57,47) ---> (None,5,57,47)
model.add(Convolution2D(nb_filter=5,nb_row=3,nb_col=3,border_mode='same',input_shape=(1,57,47)))
#(None,5,57,47) --->(None,5,28,23)
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
#(None,5,28,23) --->(None,10,28,23)
model.add(Convolution2D(nb_filter=10,nb_row=3,nb_col=3,border_mode='same'))
#(None,10,28,23)---(None,10,14,11)
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('softmax'))
model.compile(optimizer=SGD(lr=0.01,decay=1e-6,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data,train_labels,nb_epoch=10,batch_size=40,shuffle=True,verbose=1)
test_score,test_accuracy = model.evaluate(test_data,test_labels)
predictions = model.predict_clases(test_data,batch_size=10)
#混淆矩阵
print(confusion_matrix(test_labels,predictions))
#report
print(classification_report(test_labels,np.array(predictions)))
#模型训练了多久
end_time = datetime.datetime.now()
total_time = (end_time - start_time).seconds
print('total time is:',total_time)