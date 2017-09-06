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

learning_rate = 0.01
batch_size = 256

#非监督学习没有label
xs = tf.placeholder(tf.float32,[None,28*28])
#构建autoencoder网络tf.nn.tanh(-1,1),tf.nn.sigmoid(0,1),autoencoder输入图片没有进行reshape
encoder1 = tf.layers.dense(xs,128,tf.nn.relu)
encoder2 = tf.layers.dense(encoder1,64,tf.nn.relu)
encoder3 = tf.layers.dense(encoder2,10,tf.nn.relu)
#为了便于编码层的输出,encoder4不适用激活函数
encoder4 = tf.layers.dense(encoder3,2)
decoder0 = tf.layers.dense(encoder4,10,tf.nn.relu)
decoder1 = tf.layers.dense(decoder0,64,tf.nn.relu)
decoder2 = tf.layers.dense(decoder1,128,tf.nn.relu)
decoder3 = tf.layers.dense(decoder2,28*28,tf.nn.relu)

#计算loss,输出值和输入值作比较
loss = tf.losses.mean_squared_error(labels=xs,predictions=decoder3)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(5000):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        _,c = sess.run([train,loss],feed_dict={xs:batch_x})
        if step % 500 ==0:
            print('= = = = = = > > > > > >','epoch: ',int(step/500),'train loss %.4f' % c)

    #压缩的数据可视化
    encoder_result = sess.run(encoder4,feed_dict={xs:mnist.train.images})
    plt.scatter(encoder_result[:,0],encoder_result[:,1],c=mnist.train.labels,label='mnist distribution')
    plt.legend(loc='best')
    plt.title('different mnist digits shows in figure')
    plt.colorbar()
    plt.show()