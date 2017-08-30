#coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
tensorflow线性回归
"""
learning_rate = 0.01
epochs = 1000
step = 100

#读取数据
data = pd.read_table('linedata.txt',error_bad_lines=False,header=1) #pd.read_table读取txt文件
train_x = data.ix[0:100,1]
train_y = data.ix[0:100,2]
n_samples = train_x.shape[0] #均方误差n

x = tf.placeholder('float32') #placeholder接收真实值
y = tf.placeholder('float32')

#拟合参数
w = tf.Variable(np.random.randn(),name="weight") #np.random.randn()标准正态分布
b = tf.Variable(np.random.randn(),name="biases")

#构造线性模型
prediction = tf.add(tf.mul(x,w),b) #y = wx+b

#设置均方误差
cost = tf.reduce_sum(tf.pow(prediction-y,2))/(2*n_samples)
#梯度下降
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化变量
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init) #计算init
    for epoch in range(epochs):
        for (x_val,y_val) in zip(train_x,train_y):
            sess.run(train,feed_dict={x:x_val,y:y_val}) #训练
        if (epoch+1) % step == 0:
            c = sess.run(cost,feed_dict={x:train_x,y:train_y}) #计算cost
            w_value = sess.run(w)
            b_value = sess.run(b)
            print("epoch:",epoch+1,"cost=",c,"b=",b_value,"w=",w_value)
    c = sess.run(cost,feed_dict={x:train_x,y:train_y})
    w_value = sess.run(w)
    b_value = sess.run(b)
    print("the result is","cost=",c,"b=",b_value,"w=",w_value)

    #绘制训练结果
    plt.plot(train_x,train_y,'bo',label="real training data")
    plt.plot(train_x,w_value*train_x+b_value,label='fit data')
    plt.grid(True)
    plt.legend()
    plt.show()

    #测试数据
    test_x = data.ix[101:,1]
    test_y = data.ix[101:,2]

    test_cost = sess.run(tf.reduce_sum(tf.pow(y-prediction,2))/2*test_y.shape[0],feed_dict={x:test_x,y:test_y})
    #绘制测试结果
    plt.plot(test_x,test_y,'ro',label="real testing data")
    plt.plot(test_x,w_value*test_x+b_value,label='fit data')
    plt.grid(True)
    plt.legend()
    plt.show()