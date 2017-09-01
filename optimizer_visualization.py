#coding:utf-8
'''
tensorflow 1.1
python 3 
matplotlib 2.02
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#设置随机种子
tf.set_random_seed(100)
np.random.seed(100)
learning_rate = 0.01
BATCH_SIZE = 32

#构造数据
x = np.linspace(-1,1,100)[:,np.newaxis] #列向量
noise = np.random.normal(0,0.1,x.shape)
y = np.power(x,3)+noise

#构建神经网络
class Neural_network:
    def __init__(self,optimizer,**kwargs):
        self.x = tf.placeholder(tf.float32,[None,1])
        self.y = tf.placeholder(tf.float32,[None,1])
        l1 = tf.layers.dense(self.x,10,tf.nn.relu)
        output = tf.layers.dense(l1,1)
        self.loss = tf.losses.mean_squared_error(self.y,output)
        self.train = optimizer(learning_rate,**kwargs).minimize(self.loss)

#定义不同的优化器
optimizer_SGD = Neural_network(tf.train.GradientDescentOptimizer)
optimizer_Momentum = Neural_network(tf.train.MomentumOptimizer,momentum=0.9)
optimizer_RMSProp = Neural_network(tf.train.RMSPropOptimizer)
optimizer_Adam = Neural_network(tf.train.AdamOptimizer)
optimizers = [optimizer_SGD,optimizer_Momentum,optimizer_RMSProp,optimizer_Adam]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #定义一个记录losses的list
    loss_record = [[],[],[],[]]
    for step in range(300):
        #random.randint(a,b.n) #提取a-b之间n个随机整数
        index = np.random.randint(0,x.shape[0],BATCH_SIZE)
        #每次使用不同的数据训练神经网络得到losses
        train_x = x[index]
        train_y = y[index] 
        #zip()将优化器和loss组成字典
        for opti,loss_r in zip(optimizers,loss_record):         
            _,l = sess.run([opti.train,opti.loss],feed_dict={opti.x:train_x,opti.y:train_y})
            loss_r.append(l)

    #绘图
    labels = ['SGD','Momentum',"RMSProp",'Adam']
    #enumerate()枚举类型
    for i,loss_r in enumerate(loss_record):
        plt.plot(loss_r,label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.ylim((0, 0.2)) #设置y轴范围
    plt.show()  