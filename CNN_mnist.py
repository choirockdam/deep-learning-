#coding:utf-8
import tensorflow as tf
import input_data
import datetime

#程序开始执行时间
start_time = datetime.datetime.now()

learning_rate = 0.001

#加载数据集
mnist = input_data.read_data_sets('mnist/',one_hot=True)

#定义weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape,mean=0,stddev=0.1) #truncated_normal()截断的正态分布
    return tf.Variable(initial)

#定义bias
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义卷积
def conv2d(x,W):
    #strides=[1,1,1,1],第一个和最后一个默认为1,中间的1表示移动1位
    #采用same padding将边缘用0填充,可以提取边缘信息,图片大小不变
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义pooling
def max_pool_2x2(x):
    #pooling表示将图像压缩
    #ksize=[1,2,2,1],第一位和最后一位默认为1,中间2表示移动两位
    #strides=[1,2,2,1] 和 ksize相同
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#将输入神经网络的图像变形
#-1表示输入图片的个数,每张图片28行28列,灰度图1
x_image = tf.reshape(xs,[-1,28,28,1])   #输入形状[N,28,28,1]

#定义第一层卷积层
#5*5的卷积核,输入图像高度为1,输出图像高度32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#卷积+activation
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #经过卷积层输出[N,28,28,32]
#pooling
h_pool1 = max_pool_2x2(h_conv1) #经过pooling层输出 [N,14,14,32]

#定义第二层卷积层
#5*5的卷积核,图像高度由32变到64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #经过卷积层输出[N,14,14,64]
h_pool2 = max_pool_2x2(h_conv2) #经过pooling输出[N,7,7,64]

#定义全连接层
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #将pooling层输出变形[N,7*7*64]
W_fc1 = weight_variable([7*7*64,1024]) #全连接层输入7*7*64,输出1024
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#添加dropout
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#定义第二层全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#定义cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys =mnist.train.next_batch(100) #一批训练100个数据
    _,c = sess.run([train_step,cross_entropy],feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.8})
    #计算测试集合accuracy
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    accuracy = sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1})
    if i % 100 ==0:
        print("epoch",(i/100),"accuracy is",accuracy)

#程序执行完
endtime = datetime.datetime.now()
total_time = (endtime - start_time).seconds
print('total time is:',total_time,'s')