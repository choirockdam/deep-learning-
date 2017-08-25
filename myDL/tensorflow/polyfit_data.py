#coding:utf-8
"""
构造神经网络拟合数据
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#构造神经网络（正向构造）
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random.normal([in_size,out_size],mean=0,stddev=1)) 
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) 
    out1 = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = out1
    else:
        outputs = activation_function(out1)
#构造数据
x_data = np.linspace(-1,1,300)[:,np.newaxis] #列向量
noise = np.random.normal(0.1,0.05,x_data.shape)
y_data = np.power(x_data,3)+noise
xs = tf.placeholder(tf.float32,[None,1]) #列向量
ys = tf.placeholder(tf.float32,[None,1])
#构建模型
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fg = plt.gigure()
ax = fg.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() #打开交互模式
plt.show()

for i in range(5000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)