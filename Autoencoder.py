#coding:utf-8
"""
tensorflow 1.1
python 3 
matplotlib 2.02
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('mnist/',one_hot=False)
#设置随机种子
tf.set_random_seed(100)
np.random.seed(100)
learning_rate = 0.01
N_pictures = 5
batch_size = 64

#非监督学习没有label
xs = tf.placeholder(tf.float32,[None,28*28])

#构建autoencoder网络,tf.nn.tanh(-1,1),tf.nn.sigmoid(0,1)
encoder0 = tf.layers.dense(xs,128,tf.nn.tanh)
encoder1 = tf.layers.dense(encoder0,64,tf.nn.tanh)
encoder2 = tf.layers.dense(encoder1,12,tf.nn.tanh)
encoder3 = tf.layers.dense(encoder2,3)
decoder0 = tf.layers.dense(encoder3,12,tf.nn.tanh)
decoder1 = tf.layers.dense(decoder0,64,tf.nn.tanh)
decoder2 = tf.layers.dense(decoder1,128,tf.nn.tanh)
decoder3 = tf.layers.dense(decoder2,28*28,tf.nn.sigmoid)

#计算loss
loss = tf.losses.mean_squared_error(labels=xs,predictions=decoder3)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #画图,2行5列返回图和子图
    figure_,a = plt.subplots(2,N_pictures,figsize=(5,2))
    #开始交互模式
    plt.ion()
    #测试的图
    view_figures = mnist.test.images[:N_pictures]
    for i in range(N_pictures):
        #将图片reshape为28行28列显示
        a[0][i].imshow(np.reshape(view_figures[i],(28,28)))
        #清空x轴,y轴坐标
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    for step in range(10000):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        #encoder3和decoder3需要进行run
        _,encoded,decoded,c = sess.run([train,encoder3,decoder3,loss],feed_dict={xs:batch_x})
        if step % 1000 ==0:
            print('= = = = = = > > > > > >','train loss:% .4f' % c)
            #将真实的图片和autoencoder后的图片对比
            decoder_figures = sess.run(decoder3,feed_dict={xs:view_figures})
            for i in range(N_pictures):
                #清除第一行图片
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoder_figures[i],(28,28)))
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(1)
    plt.ioff() #关闭交互模式