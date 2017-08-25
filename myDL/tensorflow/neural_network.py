#coding:utf-8
"""
first neural network
"""
import tensorflow as tf
import numpy as np
#构造神经网络
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size],mean=0.0,stddev=1.0)) #标准正态分布
    biases = tf.Variable(tf.zeros([1,out_size])+0.25)
    out1 = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = out1
    else:
        outputs = activation_function(out1)
    return outputs

#构造数据
x_data = np.linspace(-1,1,500)[:,np.newaxis] 
noise = np.random.normal(0.1,0.05,x_data.shape)
y_data = np.square(x_data)-0.25+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
l1 = add_layer(xs,1,10,,activation_function = tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-y_prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i %50:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))