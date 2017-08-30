#coding:utf-8
"""
python3
tensorflow 1.1
deep convolution gan
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import input_data

mnist = input_data.read_data_sets('mnist/',one_hot=True)
#测试数据
train_data = mnist.train.images

batch_size = 32
learning_rate = 0.0002
n_input = 28*28
n_noise = 100

#discriminator网络输入图片形状
x = tf.placeholder(tf.float32,[None,n_input])
z = tf.placeholder(tf.float32,[None,n_noise])
keep_prob = tf.placeholder(tf.float32)

#定义激活函数lrelu
def lrelu(x,th=0.2):
	return tf.maximum(th*x,x)


#构建generator网络
def generator(noise_z):
	dense = tf.layers.dense(noise_z,1024)
	dense1 = tf.nn.relu(tf.layers.batch_normalization(dense))
	dense2 = tf.layers.dense(dense1,7*7*128)
	dense3 = tf.nn.relu(tf.layers.batch_normalization(dense2))
	dense4 = tf.reshape(dense3,[-1,7,7,128])
	conv1 = tf.layers.conv2d_transpose(inputs=dense4,filters=64,kernel_size=5,strides=2,padding='same')
	#batch_normalization
	relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
	conv2 = tf.layers.conv2d_transpose(relu1,filters=1,kernel_size=5,strides=2,padding='same')
	dense5 = tf.reshape(conv2,[-1,784])
	return tf.nn.sigmoid(dense5)

#构建discriminator网络
def discriminator(input_s):
	input_s = tf.reshape(input_s,[-1,28,28,1])
	conv1 = tf.layers.conv2d(inputs=input_s,filters=64,kernel_size=5,strides=2,padding='same')
	lrelu1 = lrelu(conv1,0.2)
	conv2 = tf.layers.conv2d(lrelu1,filters=128,kernel_size=5,strides=2,padding='same')
	lrelu2 = lrelu(conv2,0.2)
	flat = tf.reshape(lrelu2,[-1,7*7*128])
	dense1 = tf.layers.dense(flat,256)
	lrelu3 = lrelu(dense1,0.2)
	dropout = tf.layers.dropout(lrelu3,keep_prob)
	output = tf.layers.dense(dropout,1,activation=tf.nn.sigmoid)
	return output

#生成网络生成一张图片
generator_output = generator(z)
#判别网络根据生成网络生成的图片判别真假的概率
discriminator_pred = discriminator(generator_output)
#判别网络根据真实图片判断其为真假的概率
discriminator_real = discriminator(x)

#生成网络loss
generator_loss = tf.reduce_mean(tf.log(discriminator_pred))
#判别网络loss
discriminator_loss = tf.reduce_mean(tf.log(discriminator_real)+tf.log(1 - discriminator_pred))

generator_train = tf.train.AdamOptimizer(learning_rate).minimize(-generator_loss)
discriminator_train = tf.train.AdamOptimizer(learning_rate).minimize(-discriminator_loss)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()
	generator_c,discriminator_c = 0,0
	#开始交互模式
	#plt.ion()
	total_batch = int(mnist.train.num_examples/batch_size)
	for epoch in range(100):
		for i in range(total_batch):
			batch_x,_ = mnist.train.next_batch(batch_size)
			noise = np.random.normal(size=(batch_size,n_noise))
			_,generator_c = sess.run([generator_train,generator_loss],feed_dict={z:noise})
			_,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise,keep_prob:0.5})
		if epoch % 1 ==0:
			print('epoch:',int(epoch),'--generator_loss:%.4f'%generator_c,'--discriminator_loss:%.4f'%discriminator_c)			
	save_path = saver.save(sess,"my_net/save_net.ckpt")
"""
		#图片显示
		if epoch % 1 ==0:
			new_batch = 3
			noise = np.random.normal(size=(new_batch,n_noise))
			#生成图像
			samples = sess.run(generator_output,feed_dict={z:noise})
			fig,a = plt.subplots(1,new_batch,figsize=(new_batch*2,2))
			for i in range(new_batch):
				a[i].clear()
				a[i].set_axis_off()
				a[i].imshow(np.reshape(samples[i],(28,28)))
			plt.show()
			plt.pause(0.1)
		
		
		if epoch % 1 == 0:
			new_batch = 3
			noise = np.random.normal(size=(new_batch,n_noise))
			#生成图像
			samples = sess.run(generator_output,feed_dict={z:noise})
			fig,a = plt.subplots(1,new_batch,figsize=(new_batch,1))
			for i in range(new_batch):
				a[i].set_axis_off()
				a[i].imshow(np.reshape(samples[i],(28,28)))
			plt.savefig('samples/%i.png' %int(epoch/1))
			plt.close(fig)
"""	
	#plt.ioff()








