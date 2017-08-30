#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('mnist',one_hot=True) 

learning_rate = 0.01
epochs = 25
batch_size = 100
step = 1
cost_plt = []
accuracy_plt = []
epochs_plt = []

#正向输入
x = tf.placeholder(tf.float32,[None,784]) #None表示所有样本点
y = tf.placeholder(tf.float32,[None,10]) 
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,w)+b)  #softmax可视为分多类的logistic

#定义cost交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),reduction_indices=1))
#梯度下降
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0 #定义平均误差
        total_batch = int(mnist.train.num_examples/batch_size) #数据分为多少batch
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
            c = sess.run(cost,feed_dict={x:batch_xs,y:batch_ys})
            avg_cost += c/total_batch
        if (epoch+1) % step ==0:
            print('epoch:',(epoch+1),"cost:",c)
            cost_plt.append(c)
            epochs_plt.append(epoch+1)

        true_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #判断预测值和真实值是否相等
        accuracy = tf.reduce_mean(tf.cast(true_prediction,tf.float32)) #tf.cast将数据转为float32类型
        accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("accuracy:",accuracy)
        accuracy_plt.append(accuracy)

    #绘图
    plt.plot(epochs_plt,accuracy_plt,label="accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(epochs_plt,cost_plt,label="cost")
    plt.grid(True)
    plt.legend()
    plt.show()