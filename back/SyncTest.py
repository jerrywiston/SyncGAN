import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def class_list(imgs, labels, c=10):
	imgs_class_list = []
	for i in range(c):
		imgs_class_list.append([])
	
	for i in range(labels.shape[0]):
		imgs_class_list[labels[i]].append(imgs[i])

	return np.asarray(imgs_class_list)

def sync_match_next_batch(img1_list, img2_list, size, cut=2000):
	img1_samp = []
	img2_samp = []

	for i in range(size):
		n = random.randint(0, len(img1_list)-1)
		r = random.randint(0, cut)

		img1_samp.append(img1_list[n][r])
		img2_samp.append(img2_list[n][r])

	img1_samp_np = np.asarray(img1_samp)
	img2_samp_np = np.asarray(img2_samp)
	sync_samp_np = np.ones((size, 1))

	return img1_samp_np, img2_samp_np, sync_samp_np

def nsync_match_next_batch(img1_list, img2_list, size):
	img1_samp = []
	img2_samp = []
	
	for i in range(size):
		n1 = random.randint(0, len(img1_list)-1)
		n2 = random.randint(0, len(img2_list)-1)

		r1 = random.randint(0, len(img1_list[n1])-1)
		r2 = random.randint(0, len(img2_list[n2])-1)

		img1_samp.append(img1_list[n1][r1])
		img2_samp.append(img2_list[n2][r2])

	img1_samp_np = np.asarray(img1_samp)
	img2_samp_np = np.asarray(img2_samp)
	sync_samp_np = np.zeros((size, 1))

	return img1_samp_np, img2_samp_np, sync_samp_np

#==================== Dataset ====================
mnist_digit = input_data.read_data_sets('MNIST_digit', one_hot=False)
x_digit_train = mnist_digit.train.images
y_digit_train = mnist_digit.train.labels
x_digit_test = mnist_digit.test.images
y_digit_test = mnist_digit.test.labels
x1_train = class_list(x_digit_train, y_digit_train, 10)
x1_test = class_list(x_digit_test, y_digit_test, 10)

mnist_fashion = input_data.read_data_sets('MNIST_fashion', one_hot=False)
x_fashion_train = mnist_fashion.train.images
y_fashion_train = mnist_fashion.train.labels
x_fashion_test = mnist_fashion.test.images
y_fashion_test = mnist_fashion.test.labels
x2_train = class_list(x_fashion_train, y_fashion_train, 10)
x2_test = class_list(x_fashion_test, y_fashion_test, 10)

#==================== Util Function ====================
def xavier_init(size):
    if len(size) == 4:
        n_inputs = size[0]*size[1]*size[2]
        n_outputs = size[3]
    else:
        n_inputs = size[0]
        n_outputs = size[1]
    
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal(size, stddev=stddev)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride=[1,2,2,1]):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

#==================== Placeholder ====================
x1_ = tf.placeholder(tf.float32, shape=[None, 784])
x2_ = tf.placeholder(tf.float32, shape=[None, 784])
s_ = tf.placeholder(tf.float32, shape=[None, 1])
batch_size = 64

#==================== Variable =======================
#Mode 1
W_m1_s_conv1 = tf.Variable(xavier_init([5,5,1,16]))
b_m1_s_conv1 = tf.Variable(tf.zeros(shape=[16]))
W_m1_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m1_s_conv2 = tf.Variable(tf.zeros(shape=[32]))
W_m1_s_fc3 = tf.Variable(xavier_init([7*7*32, 256]))
b_m1_s_fc3 = tf.Variable(tf.zeros(shape=[256]))

W_m1_s_fc4 = tf.Variable(xavier_init([256,7*7*32]))
b_m1_s_fc4 = tf.Variable(tf.zeros(shape=[7*7*32]))
W_m1_s_dconv5 = tf.Variable(xavier_init([3,3,16,32]))
b_m1_s_dconv5 = tf.Variable(tf.zeros(shape=[16]))
W_m1_s_dconv6 = tf.Variable(xavier_init([5,5,1,16]))
b_m1_s_dconv6 = tf.Variable(tf.zeros(shape=[1]))

#Mode 2
W_m2_s_conv1 = tf.Variable(xavier_init([5,5,1,16]))
b_m2_s_conv1 = tf.Variable(tf.zeros(shape=[16]))
W_m2_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_s_conv2 = tf.Variable(tf.zeros(shape=[32]))
W_m2_s_fc3 = tf.Variable(xavier_init([7*7*32, 256]))
b_m2_s_fc3 = tf.Variable(tf.zeros(shape=[256]))

W_m2_s_fc4 = tf.Variable(xavier_init([256,7*7*32]))
b_m2_s_fc4 = tf.Variable(tf.zeros(shape=[7*7*32]))
W_m2_s_dconv5 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_s_dconv5 = tf.Variable(tf.zeros(shape=[16]))
W_m2_s_dconv6 = tf.Variable(xavier_init([5,5,1,16]))
b_m2_s_dconv6 = tf.Variable(tf.zeros(shape=[1]))

#Share
W_s_s4 = tf.Variable(xavier_init([512,256]))
b_s_s4 = tf.Variable(tf.zeros(shape=[256]))
W_s_s5 = tf.Variable(xavier_init([256,1]))
b_s_s5 = tf.Variable(tf.zeros(shape=[1]))

#==================== Model connect ====================
def Encoder1(x1):
	x1_re = tf.reshape(x1, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x1_re, W_m1_s_conv1, [1,2,2,1]) + b_m1_s_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_m1_s_conv2, [1,2,2,1]) + b_m1_s_conv2)
	h_re2 = tf.reshape(h_conv2, [-1,7*7*32])
	h_fc3 = tf.nn.relu(tf.matmul(h_re2, W_m1_s_fc3) + b_m1_s_fc3)
	return h_fc3

def Decoder1(v1):
	h_fc4 = tf.nn.relu(tf.matmul(v1, W_m1_s_fc4) + b_m1_s_fc4)
	h_re4 = tf.reshape(h_fc4, [-1,7,7,32])

	output_shape_g5 = tf.stack([tf.shape(v1)[0], 14, 14, 16])
	h_dconv5 = tf.nn.relu(deconv2d(h_re4, W_m1_s_dconv5, output_shape_g5) + b_m1_s_dconv5)
	
	output_shape_g6 = tf.stack([tf.shape(v1)[0], 28, 28, 32])
	x1_re_logit = deconv2d(h_dconv5, W_m1_s_dconv6, output_shape_g6) + b_m1_s_dconv6
	x1_re_prob = tf.nn.sigmoid(x1_re_logit)
	return x1_re_prob, x1_re_logit

def Encoder2(x2):
	x2_re = tf.reshape(x2, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x2_re, W_m2_s_conv1, [1,2,2,1]) + b_m2_s_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_m2_s_conv2, [1,2,2,1]) + b_m2_s_conv2)
	h_re2 = tf.reshape(h_conv2, [-1,7*7*32])
	h_fc3 = tf.nn.relu(tf.matmul(h_re2, W_m2_s_fc3) + b_m2_s_fc3)
	return h_fc3

def Decoder2(v2):
	h_fc4 = tf.nn.relu(tf.matmul(v2, W_m2_s_fc4) + b_m2_s_fc4)
	h_re4 = tf.reshape(h_fc4, [-1,7,7,32])

	output_shape_g5 = tf.stack([tf.shape(v2)[0], 14, 14, 16])
	h_dconv5 = tf.nn.relu(deconv2d(h_re4, W_m2_s_dconv5, output_shape_g5) + b_m2_s_dconv5)
	
	output_shape_g6 = tf.stack([tf.shape(v2)[0], 28, 28, 32])
	x2_re_logit = deconv2d(h_dconv5, W_m2_s_dconv6, output_shape_g6) + b_m2_s_dconv6
	x2_re_prob = tf.nn.sigmoid(x2_re_logit)
	return x2_re_prob, x2_re_logit

def Synchronizer(v1, v2):
	v = tf.concat(axis=1, values=[v1, v2])
	h_s4 = tf.nn.relu(tf.matmul(v, W_s_s4) + b_s_s4)
	s_logit = tf.matmul(h_s4, W_s_s5) + b_s_s5
	s_prob = tf.nn.sigmoid(s_logit)
	return s_prob, s_logit

V1 = Encoder1(x1_)
V2 = Encoder2(x2_)
X1_re, _ = Decoder1(V1)
X2_re, _ = Decoder2(V2)
S_prob, S_logit = Synchronizer(V1, V2)

#==================== Loss and Train ====================
E1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(X1_re - x1_), reduction_indices=[1]))
E2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(X2_re - x2_), reduction_indices=[1]))
Ss_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_logit, labels=s_))
Ss_solver = tf.train.AdamOptimizer().minimize(Ss_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#==================== Main ====================
for it in range(40001):
	x1_sync, x2_sync, s_sync = sync_match_next_batch(x1_train, x2_train, batch_size)
	x1_nsync, x2_nsync, s_nsync = nsync_match_next_batch(x1_train, x2_train, batch_size)
	
	x1_batch = np.concatenate((x1_sync, x1_nsync), axis=0)
	x2_batch = np.concatenate((x2_sync, x2_nsync), axis=0)
	sr_batch = np.concatenate((s_sync, s_nsync), axis=0)

	_, loss_ss = sess.run([Ss_solver, Ss_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, s_:sr_batch})
	if it%100 == 0:
		print("Iter: {}, loss:{:.4}".format(it, loss_ss))