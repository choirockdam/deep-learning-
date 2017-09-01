#coding:utf-8
"""
tensorflow 1.1
python 3
matplotlib 2.02
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) #随机种子
tf.set_random_seed(100) #设置随机种子

a = np.ones((500,2))
noise = np.random.normal(4,1,(500,2))
x0 = np.random.normal(2*a,1)+noise #正态分布,标砖差为1
x1 = np.random.normal(7*a,1)+noise 
x = np.vstack((x0,x1)) #垂直合并 (200,2)
y0 = np.zeros(500)
y1 = np.ones(500)
y = np.hstack((y0,y1)) #水平合并
xs = tf.placeholder(tf.float32,x.shape)  #输入形状(200,2)
ys = tf.placeholder(tf.int32,y.shape) #输出形状(200,) 

#构建神经网络
l1 = tf.layers.dense(xs,50,tf.nn.relu)
output = tf.layers.dense(l1,2)
#定义损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=ys,logits=output)
#定义计算准确度函数
#tf.metrics.accuracy计算精度,返回accuracy和update_operation
accuracy = tf.metrics.accuracy(labels=ys,predictions=tf.argmax(output,axis=1))[1]
#梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
with tf.Session() as sess:
    #初始化
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)
    #打开交互模式
    plt.ion()
    for step in range(1000):
        _,acc,pred = sess.run([optimizer,accuracy,output],feed_dict={xs:x,ys:y})
        if step % 50 == 0:
            plt.clf() #清空当前图像
            plt.scatter(x[:,0],x[:,1],c = pred.argmax(1),s=100,marker='*',cmap='RdYlGn') #cmap是画布
            plt.text(2,12,'accuracy=%.2f' %acc,fontdict={'size':15,'color':'red'})
            plt.pause(0.1) #暂停
    plt.ioff() #关闭交互模式
    plt.show()