#coding:utf-8

"""
tensorflow 1.1
matplotlib 2.02
python3

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#设置随机种子
tf.set_random_seed(100)
np.random.seed(100)

#数据
x = np.linspace(-1,1,100)[:,np.newaxis] #列向量
noise = np.random.normal(0,0.15,x.shape)
y = np.power(x,3)+noise

def save_function():
    print('this is save')
    xs = tf.placeholder(tf.float32,x.shape)
    ys = tf.placeholder(tf.float32,y.shape)
    #建立神经网络
    l1 = tf.layers.dense(xs,10,tf.nn.relu)
    output = tf.layers.dense(l1,1)
    #定义损失函数
    loss = tf.losses.mean_squared_error(ys,output)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
    with tf.Session() as sess:
        #初始化
        init = tf.global_variables_initializer()
        sess.run(init)
        #定义一个saver
        saver = tf.train.Saver()
        for step in range(100):
            sess.run(optimizer,feed_dict={xs:x,ys:y})
        #saver.save()
        saver.save(sess,'params',write_meta_graph=False)
        pred,l = sess.run([output,loss],feed_dict={xs:x,ys:y})
        #绘图
        plt.figure(1,figsize=(10,5))
        plt.subplot(121)
        plt.scatter(x,y)
        plt.plot(x,pred,color='salmon',linestyle='--',lw=5)
        plt.text(0,0.5,'loss=%.4f' %l,fontdict={'size':15,'color':'salmon'})
        plt.title('save loss graph')

def reload_function():
    print('this is reload')
    xs = tf.placeholder(tf.float32,x.shape)
    ys = tf.placeholder(tf.float32,y.shape)
    l1 = tf.layers.dense(xs,10,tf.nn.relu)
    output = tf.layers.dense(l1,1)
    loss = tf.losses.mean_squared_error(ys,output)
    #不需要定义优化器和初始化
    with tf.Session() as sess:
        saver = tf.train.Saver() #定义一个saver还原参数
        saver.restore(sess,'params') #saver.restore()还原,不需要定义optimizer和初始化变量
        #画图
        pred,l = sess.run([output,loss],feed_dict={xs:x,ys:y})
        plt.subplot(1,2,2)
        plt.scatter(x,y)
        plt.plot(x,pred,color='c',linestyle='-',lw=5)
        plt.text(0,0.5,'loss=%.4f' %l,fontdict={'size':15,'color':'c'})
        plt.title('reload loss graph')
        plt.show()

save_function()
#重置之前建立的图
tf.reset_default_graph()
#重新加载
reload_function()