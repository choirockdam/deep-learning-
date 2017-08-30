#coding:utf-8
"""
构造神经网络结果可视化
"""
import tensorflow as tf
import numpy as np
#构造神经网络
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = "layer %s" %n_layer
    with tf.name_scope(layer_name): #结点名称
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size],mean=1,stddev=0),name="w")
            tf.histogram_summary(layer_name+'/weights',Weights) #绘图
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]+0.25))
            tf.histogram_summary(layer_name+'/biases',biases)
        with tf.name_scope('out1'):
            out1 = tf.matmul(inputs,Weights)+biases
            tf.histogram_summary(layer_name+'/out1',out1)
        if activation_function is None:
            outputs = out1
        else:
            outputs = activation_function(out1)
        tf.histogram_summary(layer_name+'/outputs',outputs)
        return outputs
#构造数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0.1,0.25,x_data.shape)
y_data = np.square(x_data)+noise

with tf.scope_name('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name == 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name == 'y_input')

l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),prediction_indices=[1]))
    tf.scalar_summary("loss",loss) #loss用scalar_summary
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #梯度下降

init = tf.initialize_all_variables()
sess = tf.Session()

merged = tf.merge_all_summarizes() #将所有统计信息汇聚
writer = tf.train.SummaryWriter('logs/',sess.graph) #写入log目录
sess.run(init)

for i in range(5000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50:
        resule = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)