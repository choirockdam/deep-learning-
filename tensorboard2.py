#coding:utf-8
"""
python 3
tensorflow 1.1
"""
import tensorflow as tf
import numpy as np

#设置随机种子
tf.set_random_seed(100)
np.random.seed(100)

#数据
x = np.linspace(-1,1,100)[:,np.newaxis]
noise = np.random.normal(0,0.1,x.shape)
y = np.power(x,3) + noise

#输入可视化(tf.name_scope())
with tf.variable_scope('inputs'):
    xs = tf.placeholder(tf.float32,x.shape,name='x')
    ys = tf.placeholder(tf.float32,y.shape,name='y')

#神经网络可视化
with tf.variable_scope('neural_network'):
    l1 = tf.layers.dense(xs,10,tf.nn.relu,name='hidden_layer')
    output = tf.layers.dense(l1,1,name='output_layer')
    #变量值统计
    tf.summary.histogram('layer1',l1)
    tf.summary.histogram('output',output)

#计算误差scope = 'loss'
loss = tf.losses.mean_squared_error(y,output,scope='loss')
#梯度下降
train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
#loss的统计用tf.summary.scalar()
tf.summary.scalar('loss',loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #tf.summary.merge_all()将所有统计信息合并(tf.summary.histogram,tf.summary_scalar)
    merged = tf.summary.merge_all()
    #tf.summary.FileWriter()将所有信息写入文件
    writer = tf.summary.FileWriter('./tensorflow1.1_logs',sess.graph)   
    for step in range(100):
        #merged也要训练
        _,result = sess.run([train,merged],feed_dict={xs:x,ys:y})
        writer.add_summary(result,step)