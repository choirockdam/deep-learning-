#coding:utf-8
"""
python3
tensorflow 1.1
matplotlib 2.02
change file
"""
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

#读取数据集
with open('facedataset.pickle','rb') as f:
    (train_data,train_labels),(test_data,test_labels) = pickle.load(f)

learning_rate = 0.01
N_pictures = 3


train_data = train_data.astype(np.float32)
train_data = np.random.permutation(train_data)
test_data = test_data.astype(np.float32)
test_data = np.random.permutation(test_data)

xs = tf.placeholder(tf.float32,[None,57*47])

#构建autoencoder神经网络
encoder0 = tf.layers.dense(xs,512,tf.nn.tanh)
encoder1 = tf.layers.dense(encoder0,128,tf.nn.tanh)
encoder2 = tf.layers.dense(encoder1,64,tf.nn.tanh)
encoder3 = tf.layers.dense(encoder2,10)
decoder0 = tf.layers.dense(encoder3,64,tf.nn.tanh)
decoder1 = tf.layers.dense(decoder0,128,tf.nn.tanh)
decoder2 = tf.layers.dense(decoder1,512,tf.nn.tanh)
decoder3 = tf.layers.dense(decoder2,57*47,tf.nn.sigmoid)

#计算loss
loss = tf.losses.mean_squared_error(labels=xs,predictions=decoder3)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #画图plt.subplots
    fig,a = plt.subplots(2,N_pictures,figsize=(6,4))
    #开始交互模式
    plt.ion()
    view_figures = test_data[:N_pictures]
    for i in range(N_pictures):
        #将测试的真实的图显示
        a[0][i].imshow(np.reshape(view_figures[i],(57,47)))
        #清空坐标轴
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    for step in range(1000):
        _,encoderd,decodered,c = sess.run([train,encoder3,decoder3,loss],feed_dict={xs:train_data})
        if step % 100 ==0:
            print('= = = = = = > > > > > > step:',int(step/100),'train loss: %.4f'%c)
        #将测试集中真实图片作为预测的图片
        decoder_figures = sess.run(decoder3,feed_dict={xs:view_figures})
        for i in range(N_pictures):
            #清除第二行第i张图片
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoder_figures[i],(57,47)))
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.show()
        plt.pause(1)
        #关闭交互模式
    plt.ioff()