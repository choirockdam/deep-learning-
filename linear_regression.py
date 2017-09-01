#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1,500)[:,np.newaxis] #列向量
noise = np.random.normal(0,0.1,x.shape)
y = np.power(x,3) + noise

xs = tf.placeholder(tf.float32,x.shape)
ys = tf.placeholder(tf.float32,y.shape)

#构建神经网络
#输入,输出神经元个数,激活函数
l1 = tf.layers.dense(xs,20,tf.nn.relu) #输出10个神经元的隐藏层,激活函数relu
output = tf.layers.dense(l1,1) #输入l1,输出神经元个数1

#定义均方误差loss
#tf.losses.mean_squared_error
loss = tf.losses.mean_squared_error(ys,output) #均方误差
#定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.4).minimize(loss) #数据量较小调大learning_rate使其学习加快

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    plt.ion() #打开交互模式
    for step in range(100):
        _,c = sess.run([optimizer,loss],feed_dict={xs:x,ys:y})
        prediction = sess.run(output,feed_dict={xs:x}) #计算预测值
        if step % 5 == 0:
            #可以用clf()来清空当前图像，用cla()来清空当前坐标
            plt.clf()#清空当前图像
            plt.scatter(x,y)
            plt.plot(x,prediction,'c-',lw='5')
            plt.text(0,0.5,'cost=%.4f' % c,fontdict={'size':15,'color':'red'}) #添加text,位置在坐标轴0,0.5处
            plt.pause(0.1) #暂停0.1s
    plt.ioff() #关闭交互模式
    plt.show(