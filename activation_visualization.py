#coding:utf-8
"""
tensorflow 1.1
python 3
matplotlib 2.02
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,400)

#定义激活函数
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

with tf.Session() as sess:
    [y_relu,y_sigmoid,y_softplus,y_tanh] = sess.run([y_relu,y_sigmoid,y_softplus,y_tanh])
    #画图
    plt.figure(1,figsize=(8,6))
    plt.subplot(221)
    plt.plot(x, y_relu, c='blue', label='relu')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(x, y_sigmoid, c='red', label='sigmoid')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(x, y_tanh, c='c', label='tanh')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(x, y_softplus, c='yellow', label='softplus')
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()  