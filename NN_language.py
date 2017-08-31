#coding:utf-8
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

#从pickle文件读取数据
pkl_file = open('sentiment_set.pickle', 'rb')
train_x, train_y, test_x, test_y = pickle.load(pkl_file)

n_nodes_hl1 = 128
n_nodes_hl2 = 256
n_nodes_hl3 = 512
n_classes = 2
batch_size = 100
learning_rate=0.01
epoch_plt = []
accuracy_plt = []
cost_plt = []


x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

#构建神经网络
def neural_network_model(data):

    hidden_1_layer = {
                      'weights': tf.Variable(tf.random_normal( [len(train_x[0]), n_nodes_hl1],mean=0,stddev=1)),
                      'biases': tf.Variable(tf.zeros([n_nodes_hl1]))
                      }

    hidden_2_layer = {
                      'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2],mean=0,stddev=1)),
                      'biases': tf.Variable(tf.zeros([n_nodes_hl2]))
                      }

    hidden_3_layer = {
                      'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3],mean=0,stddev=1)),
                      'biases': tf.Variable(tf.zeros([n_nodes_hl3]))
                      }

    output_layer = {
                       'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes],mean=0,stddev=1)),
                       'biases': tf.Variable(tf.zeros([n_classes]))
                    }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #初始化变量
        for epoch in range(epochs):
            i = 0
            while i < len(train_x):
                #按批进行训练
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                train = sess.run(optimizer,feed_dict={x: batch_x, y: batch_y})
                c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                i += batch_size
            cost_plt.append(c)
            epoch_plt.append(epoch+1)
            print('Epoch', epoch + 1, 'loss:', c)
            #比较预测值和实际值是否相同,1表示按照 1轴方向
            correct = tf.equal(tf.arg_max(prediction, 1), tf.argmax(y, 1))
            #将布尔值转为float类型后求平均
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            accuracy = sess.run(accuracy,feed_dict={x: test_x, y: test_y})
            accuracy_plt.append(accuracy)
            print('Accuracy:',accuracy)

        plt.plot(epoch_plt,accuracy_plt,'r',lw=2,label="accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.plot(epoch_plt,cost_plt,'c',lw=2,label="cost")
        plt.grid(True)
        plt.legend()
        plt.show()

train_neural_network(x)