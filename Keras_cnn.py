#coding:utf-8
"""
python 2.7
keras 2.0.4
"""
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import input_data
import datetime

start_time = datetime.datetime.now()
#设置随机种子
np.random.seed(1000)

#数据格式转换
#one_hot=False这里故意使y值为如下表示：(0000000000),目的是后面使用keras的np_utils
mnist = input_data.read_data_sets('mnist/',one_hot=False)
#样本数,颜色通道,28行28列
train_data=mnist.train.images.reshape(mnist.train.images.shape[0],1,28,28)
#通过keras的np_utils将y值转为如下表示：(0000000000)
train_labels = np_utils.to_categorical(mnist.train.labels,nb_classes=10)
test_data = mnist.test.images.reshape(mnist.test.images.shape[0],1,28,28)
test_labels = np_utils.to_categorical(mnist.test.labels,nb_classes=10)

#构建模型
model = Sequential()
#卷积层,32个卷积核,每个卷积核大小5*5,采用same_padding的方式
model.add(Convolution2D(nb_filter=32,nb_row=5,nb_col=5,border_mode='same',input_shape=(1,28,28)))
#pooling层,采用same padding 
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
model.add(Convolution2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
#将数据展平
model.add(Flatten())
#全连接层
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
#编译模型sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)  
model.compile(optimizer=Adam(lr = 0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#训练模型
#shuffle就是是否把数据随机打乱之后再进行训练  
# verbose是屏显进度条   
# validation_split就是拿出百分之多少用来做交叉验证  
model.fit(train_data,train_labels,nb_epoch=10,batch_size=50,shuffle=True,verbose=1,validation_split=0.3)

#测试集结果
c,acc = model.evaluate(test_data,test_labels,batch_size=50)
#输出预测分类是0,1,2,3,4,5这种类型
predictions = model.predict_classes(test_data,batch_size=50)
#混淆矩阵
print(confusion_matrix(mnist.test.labels,predictions))
#report
print(classification_report(mnist.test.labels,np.array(predictions)))
#模型训练了多久
end_time = datetime.datetime.now()
total_time = (end_time - start_time).seconds
print('total time is:',total_time)