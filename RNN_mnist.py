#coding:utf-8
"""
tensorflow 1.1
python 3 
matplotlib 2.02
"""
import tensorflow as tf
import input_data
import numpy as np
import matplotlib.pyplot as plt

#设置随机种子
tf.set_random_seed(100)
np.random.seed(100)
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
learning_rate = 0.01

mnist = input_data.read_data_sets('mnist/',one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]
#查看图片
plt.imshow(mnist.test.images[3].reshape((28,28)))
plt.title('the picture is %i' %np.argmax(mnist.test.labels[3]),fontdict={'size':16,'color':'red'})
plt.show()

xs = tf.placeholder(tf.float32,[None,TIME_STEP*INPUT_SIZE])
ys = tf.placeholder(tf.int32,[None,10])
#输入神经网络前将形状(None,28*28) --->(None,28,28)
x = tf.reshape(xs,[-1,TIME_STEP,INPUT_SIZE])

#构建循环神经网络
#tf.contrib.rnn.BasicLSTMCell()构建循环神经网络的cell
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
#tf.nn.dynamic_rnn返回outputs和states,其中states包含主要state和次要state
#输入神经网络的形状(batch,time_step,input)时间参数不在第一个维度,所以time_major=False
outputs,states = tf.nn.dynamic_rnn(rnn_cell,x,initial_state=None,dtype=tf.float32,time_major=False)
#将最后一个time_step的输出作为输出
output = tf.layers.dense(outputs[:,-1,:],10)

#计算loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=output)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#计算accuracy,返回两个值acc和uodate_op
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys,axis=1),predictions=tf.argmax(output,axis=1))[1]
with tf.Session() as sess:
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
        _,c = sess.run([train,loss],feed_dict={xs:batch_x,ys:batch_y})
        if step % 100 == 0:
            acc = sess.run(accuracy,feed_dict={xs:test_x,ys:test_y})
            print('= = = = = = > > > > > >','epoch: ',int(step/100),'train loss : %.4f' %c,'test accuracy: %.3f' %acc)