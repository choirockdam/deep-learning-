#coding:utf-8
import tensorflow as tf
import input_data

#加载mnist数据集
mnist = input_data.read_data_sets('mnist/',one_hot=True)

#定义参数
learning_rate = 0.01
epochs = 50000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

xs = tf.placeholder(tf.float32,[None,n_steps,n_inputs]) #输入形状
ys = tf.placeholder(tf.float32,[None,n_classes])

#定义weights和biases
weights = {
    "in":tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    "out":tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases = {
    "in":tf.Variable(tf.constant(0.1,shape = [1,n_hidden_units])),
    "out":tf.Variable(tf.constant(0.1,shape = [1,n_classes]))
}

def RNN(inputs,weights,biases):
    #将输入(128,28,28)维度变换(batch_size,n_inputs,n_steps)
    x = tf.reshape(inputs,[-1,n_inputs]) #x(128*28,28)
    x_in = tf.matmul(x,weights['in'])+biases['in'] #x_in(128*28,128)
    #将数据维度变换
    x_in = tf.reshape(x_in,[-1,n_steps,n_hidden_units]) #x_in(128,28,128)
    #定义RNN的cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0) #设置初始biases = 1
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32) #初始化state
    #计算RNN
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=_init_state,time_major=False)
    #输出
    outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
    results = tf.matmul(outputs[-1],weights['out'])+biases['out']
    return results

prediction = RNN(xs,weights,biases)
#计算softmax层的cross_entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,ys))
#梯度下降
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

true_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(true_pred,tf.float32))

#初始化
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < epochs:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
        if step % 50 ==0:
            print(sess.run(cost,cost,feed_dict={xs:batch_xs,ys:batch_ys}))
        step += 1
    print(sess.run(accuracy,cost,feed_dict={xs:batch_xs,ys:batch_ys}))