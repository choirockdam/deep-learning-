#coding:utf-8
import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt

"""
利用多层感知器对mnist数据集分类
精度比logistic高
"""
#加载数据
mnist = input_data.read_data_sets('mnist',one_hot="True")

#定义参数
learning_rate = 0.01
epochs = 15
batch_size = 100
step = 1

epoch_plt=[]
accuracy_plt=[]
cost_plt=[]

#定义多层感知神经网络
def multilayer_perception(inputs,in_size,out_size,activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size],mean=0,stddev=1))
    biases = tf.Variable(tf.zeros([1,out_size]))
    out1 = tf.matmul(inputs,weights)+biases
    if activation_function is None:
        output = out1
    else:
        output = activation_function(out1)
    return output

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

l1 = multilayer_perception(xs,784,256,activation_function=tf.nn.relu) #隐藏层
l2 = multilayer_perception(l1,256,256,activation_function=tf.nn.relu)
prediction = multilayer_perception(l2,256,10,activation_function=None) 

#计算最后一层是softmax层的cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = ys))
#使用AdamOptimizer进行梯度下降
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#初始化变量
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={xs:batch_x,ys:batch_y})
            c = sess.run(cost,feed_dict={xs:batch_x,ys:batch_y})
            avg_cost += c/total_batch
        if(epoch+1) % step ==0:
            print("epoch",(epoch+1),"cost=",avg_cost)
            epoch_plt.append(epoch+1)
            cost_plt.append(c)

        true_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
        accuracy = tf.reduce_mean(tf.cast(true_prediction,"float32"))
        accuracy = sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels})
        print ("Accuracy",accuracy)
        accuracy_plt.append(accuracy)

    #绘图
    plt.plot(epoch_plt,accuracy_plt,'r',lw=2,label="accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(epoch_plt,cost_plt,'c',lw=2,label="cost")
    plt.grid(True)
    plt.legend()
    plt.show()