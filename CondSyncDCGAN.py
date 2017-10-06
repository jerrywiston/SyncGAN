import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import math
import scipy.ndimage.interpolation

#==================== Draw Figure ====================
def plot(samples, size):
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(size[0], size[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def plot_x(id, type, samp, size=(4,4)):
    fig = plot(samp, size)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def samp_fig(sess, size, x1_train, x2_train):
	x_samp = np.zeros([size[0], size[1], 784], dtype=np.float32)

	for i in range(int(size[0]/2)):
		x1_sync, x2_sync, s_sync = sync_match_next_batch(x1_train, x2_train, 6)
		x_samp[i*2] = x1_sync
		x_samp[i*2+1] = sess.run(G_sample, feed_dict={x1_: x1_sync})

	x_samp = x_samp.reshape(size[0]*size[1], 784)
	return x_samp

#==================== Data Batch ====================
def class_list(imgs, labels, c=10):
	imgs_class_list = []
	for i in range(c):
		imgs_class_list.append([])
	
	for i in range(labels.shape[0]):
		imgs_class_list[labels[i]].append(imgs[i])

	return np.asarray(imgs_class_list)

def next_batch(imgs, size):
    img_samp = np.ndarray(shape=(size, imgs.shape[1]))
    for i in range(size):
        r = random.randint(0,imgs.shape[0]-1)
        img_samp[i] = imgs[r]
    return img_samp

def sync_next_batch(img1_list, img2_list, size):
	img1_samp = []
	img2_samp = []

	for i in range(size):
		n = random.randint(0, len(img1_list)-1)
		r1 = random.randint(0, len(img1_list[n])-1)
		r2 = random.randint(0, len(img2_list[n])-1)

		img1_samp.append(img1_list[n][r1])
		img2_samp.append(img2_list[n][r2])

	img1_samp_np = np.asarray(img1_samp)
	img2_samp_np = np.asarray(img2_samp)
	sync_samp_np = np.ones((size, 1))

	return img1_samp_np, img2_samp_np, sync_samp_np

def nsync_next_batch(img1_list, img2_list, size):
	img1_samp = []
	img2_samp = []
	
	for i in range(size):
		n1 = random.randint(0, len(img1_list)-1)
		n2 = random.randint(0, len(img2_list)-1)
		while n1 == n2:
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

def sample_z(m, n, type=1):
    if type == 0:
    	return np.random.uniform(-1., 1., size=[m, n])
    else:
    	return np.random.normal(0., 1., size=[m, n])

#==================== Parameter ====================
batch_size = 64

def xavier_init(size):
    if len(size) == 4:
        n_inputs = size[0]*size[1]*size[2]
        n_outputs = size[3]
    else:
        n_inputs = size[0]
        n_outputs = size[1]
    
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal(size, stddev=stddev)

def conv2d(x, W, stride, bn=True):
    if bn:
        x = tf.layers.batch_normalization(x, training=True)
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride=[1,2,2,1], bn=True):
    if bn:
        x = tf.layers.batch_normalization(x, training=True)
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

#==================== Placeholder ====================
x1_ = tf.placeholder(tf.float32, shape=[None, 784])
x2_ = tf.placeholder(tf.float32, shape=[None, 784])
s_ = tf.placeholder(tf.float32, shape=[None, 1])

#==================== Generator ====================
#Generator
W_g_conv1 = tf.Variable(xavier_init([5,5,1,32]))
b_g_conv1 = tf.Variable(tf.zeros(shape=[32]))

W_g_conv2 = tf.Variable(xavier_init([5,5,32,64]))
b_g_conv2 = tf.Variable(tf.zeros(shape=[64]))

W_g_fc3 = tf.Variable(xavier_init([7*7*64, 7*7*64]))
b_g_fc3 = tf.Variable(tf.zeros(shape=[7*7*64]))

W_g_dconv4 = tf.Variable(xavier_init([5,5,32,64]))
b_g_dconv4 = tf.Variable(tf.zeros(shape=[32]))

W_g_dconv5 = tf.Variable(xavier_init([5,5,1,32]))
b_g_dconv5 = tf.Variable(tf.zeros(shape=[1]))

var_g = [W_g_conv1, b_g_conv1, 
		 W_g_conv2, b_g_conv2,
		 W_g_fc3, b_g_fc3,
		 W_g_dconv4, b_g_dconv4, 
		 W_g_dconv5, b_g_dconv5]

def Generator(x):
	x_re = tf.reshape(x, [-1,28,28,1])
	h_g_conv1 = tf.nn.relu(conv2d(x_re, W_g_conv1, [1,2,2,1], bn=False) + b_g_conv1)
	h_g_conv2 = tf.nn.relu(conv2d(h_g_conv1, W_g_conv2, [1,2,2,1]) + b_g_conv2)
	
	h_g_re2 = tf.reshape(h_g_conv2, [-1,7*7*64])
	h_g_re2 = tf.layers.batch_normalization(h_g_re2, training=True)
	h_g_fc3 = tf.nn.relu(tf.matmul(h_g_re2, W_g_fc3) + b_g_fc3)
	h_g_re3 = tf.reshape(h_g_fc3, [-1,7,7,64])

	output_shape_g4 = tf.stack([tf.shape(x)[0], 14, 14, 32])
	h_g_dconv4 = tf.nn.relu(deconv2d(h_g_re3, W_g_dconv4, output_shape_g4) + b_g_dconv4)
	output_shape_g5 = tf.stack([tf.shape(x)[0], 28, 28, 1])
	h_g_dconv5 = tf.nn.sigmoid(deconv2d(h_g_dconv4, W_g_dconv5, output_shape_g5) + b_g_dconv5)

	h_g_re5 = tf.reshape(h_g_dconv5, [-1,784])
	return h_g_re5

#==================== Discriminator ====================
#Discriminator
W_d_conv1 = tf.Variable(xavier_init([5,5,1,16]))
b_d_conv1 = tf.Variable(tf.zeros(shape=[16]))

W_d_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_d_conv2 = tf.Variable(tf.zeros(shape=[32]))

W_d_fc3 = tf.Variable(xavier_init([7*7*32, 256]))
b_d_fc3 = tf.Variable(tf.zeros(shape=[256]))

W_d_fc4 = tf.Variable(xavier_init([256, 1]))
b_d_fc4 = tf.Variable(tf.zeros(shape=[1]))

var_d = [W_d_conv1, b_d_conv1, 
		 W_d_conv2, b_d_conv2, 
		 W_d_fc3, b_d_fc3, 
		 W_d_fc4, b_d_fc4]

def Discriminator(x):
	x_re = tf.reshape(x, [-1,28,28,1])
	h_d_conv1 = tf.nn.relu(conv2d(x_re, W_d_conv1, [1,2,2,1], bn=False) + b_d_conv1)

	h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_d_conv2, [1,2,2,1]) + b_d_conv2)
	h_d_re2 = tf.reshape(h_d_conv2, [-1,7*7*32])

	h_d_fc3 = tf.nn.relu(tf.matmul(h_d_re2, W_d_fc3) + b_d_fc3)
	
	y_logit = tf.matmul(h_d_fc3, W_d_fc4) + b_d_fc4
	y_prob = tf.nn.sigmoid(y_logit)
	
	return y_prob, y_logit

#==================== Synchronizer ====================
#Mode 1
W_m1_s_conv1 = tf.Variable(xavier_init([5,5,1,16]))
b_m1_s_conv1 = tf.Variable(tf.zeros(shape=[16]))
W_m1_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m1_s_conv2 = tf.Variable(tf.zeros(shape=[32]))
W_m1_s_fc3 = tf.Variable(xavier_init([7*7*32, 256]))
b_m1_s_fc3 = tf.Variable(tf.zeros(shape=[256]))

#Mode 2
W_m2_s_conv1 = tf.Variable(xavier_init([5,5,1,16]))
b_m2_s_conv1 = tf.Variable(tf.zeros(shape=[16]))
W_m2_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_s_conv2 = tf.Variable(tf.zeros(shape=[32]))
W_m2_s_fc3 = tf.Variable(xavier_init([7*7*32, 256]))
b_m2_s_fc3 = tf.Variable(tf.zeros(shape=[256]))

#Shared
W_s_s4 = tf.Variable(xavier_init([512,256]))
b_s_s4 = tf.Variable(tf.zeros(shape=[256]))
W_s_s5 = tf.Variable(xavier_init([256,1]))
b_s_s5 = tf.Variable(tf.zeros(shape=[1]))

var_s = [ W_m1_s_conv1, b_m1_s_conv1, 
		  W_m1_s_conv2, b_m1_s_conv2,
		  b_m1_s_fc3, b_m1_s_fc3,
		  
		  W_m2_s_conv1, b_m2_s_conv1,
		  W_m2_s_conv2, b_m2_s_conv2,
		  b_m2_s_fc3, b_m2_s_fc3,

		  W_s_s4, b_s_s4,
		  W_s_s5, b_s_s5 ]

def Synchronizer(x1, x2):
	#Mode 1
	x1_re = tf.reshape(x1, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x1_re, W_m1_s_conv1, [1,2,2,1], bn=False) + b_m1_s_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_m1_s_conv2, [1,2,2,1]) + b_m1_s_conv2)
	h_re2 = tf.reshape(h_conv2, [-1,7*7*32])
	v1 = tf.nn.relu(tf.matmul(h_re2, W_m1_s_fc3) + b_m1_s_fc3)

	#Mode 2
	x2_re = tf.reshape(x2, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x2_re, W_m2_s_conv1, [1,2,2,1], bn=False) + b_m2_s_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_m2_s_conv2, [1,2,2,1]) + b_m2_s_conv2)
	h_re2 = tf.reshape(h_conv2, [-1,7*7*32])
	v2 = tf.nn.relu(tf.matmul(h_re2, W_m2_s_fc3) + b_m2_s_fc3)

	#Shared
	v = tf.concat(axis=1, values=[v1, v2])
	h_s4 = tf.nn.relu(tf.matmul(v, W_s_s4) + b_s_s4)
	s_logit = tf.matmul(h_s4, W_s_s5) + b_s_s5
	s_prob = tf.nn.sigmoid(s_logit)
	return s_prob, s_logit

#==================== Node Connect ====================
G_sample = Generator(x1_)

D_real_prob, D_real_logit = Discriminator(x2_)
D_fake_prob, D_fake_logit = Discriminator(G_sample)

S_real_prob, S_real_logit = Synchronizer(x1_, x2_)
S_fake_prob, S_fake_logit = Synchronizer(x1_, G_sample)

#==================== Loss & Train ====================
#Vanilla GAN Loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logit, labels=tf.ones_like(D_real_logit)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)))
D_loss = D_loss_real + D_loss_fake 
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)))

#Synchronize Loss
Ss_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_real_logit, labels=s_))
Ss_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=tf.zeros_like(S_fake_logit)))
Ss_loss = Ss_loss_real + Ss_loss_fake
Gs_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=tf.ones_like(S_fake_logit)))

#Reconstruct Loss
#R_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(G_sample - x2_), reduction_indices=[1]))
R_loss = tf.reduce_mean(tf.reduce_sum(tf.square(G_sample - x2_), reduction_indices=[1]))
R_solver = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(R_loss, var_list=var_g)

#Solver 
G_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(G_loss, var_list=var_g)
D_solver = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(D_loss, var_list=var_d)

Gs_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(Gs_loss, var_list=var_g)
Ss_solver = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(Ss_loss, var_list=var_s) #fashion:2e-4, 90-degree:1e-4

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#==================== Dataset ====================
mnist_digit = input_data.read_data_sets('MNIST_digit', one_hot=False)
x_digit_train = mnist_digit.train.images
y_digit_train = mnist_digit.train.labels
x_digit_test = mnist_digit.test.images
y_digit_test = mnist_digit.test.labels
x1_train = class_list(x_digit_train, y_digit_train, 10)
x1_test = class_list(x_digit_test, y_digit_test, 10)
'''
mnist_fashion = input_data.read_data_sets('MNIST_fashion', one_hot=False)
x_fashion_train = mnist_fashion.train.images
y_fashion_train = mnist_fashion.train.labels
x_fashion_test = mnist_fashion.test.images
y_fashion_test = mnist_fashion.test.labels
x2_train = class_list(x_fashion_train, y_fashion_train, 10)
x2_test = class_list(x_fashion_test, y_fashion_test, 10)
'''
#Rotatate digit (cross domain)
x_digit_train_rot = scipy.ndimage.interpolation.rotate(x_digit_train.reshape(-1, 28, 28), 90, axes=(1, 2)).reshape(-1, 28*28)
x_digit_test_rot = scipy.ndimage.interpolation.rotate(x_digit_test.reshape(-1, 28, 28), 90, axes=(1, 2)).reshape(-1, 28*28)
x2_train = class_list(x_digit_train_rot, y_digit_train, 10)
x2_test = class_list(x_digit_test_rot, y_digit_test, 10)

#==================== Main ====================
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(40001):
	#Get batch training data
	x1_sync, x2_sync, s_sync = sync_match_next_batch(x1_train, x2_train, batch_size)
	x1_nsync, x2_nsync, s_nsync = nsync_match_next_batch(x1_train, x2_train, batch_size)
	
	x1_batch = np.concatenate((x1_sync, x1_nsync), axis=0)
	x2_batch = np.concatenate((x2_sync, x2_nsync), axis=0)
	s_batch = np.concatenate((s_sync, s_nsync), axis=0)
	
	#Training
	_, loss_d = sess.run([D_solver, D_loss], feed_dict={x1_:x1_batch, x2_:x2_batch})
	_, loss_ss = sess.run([Ss_solver, Ss_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, s_:s_batch})
	
	_, loss_g = sess.run([G_solver, G_loss], feed_dict={x1_:x1_batch})
	_, loss_gs = sess.run([Gs_solver, Gs_loss], feed_dict={x1_:x1_batch})

	#_, loss_r = sess.run([R_solver, R_loss], feed_dict={x1_:x1_sync, x2_:x2_sync})
		
	#Show result
	if it%100 == 0:
		print("Iter: {}".format(it))
		print("  G_loss : {:.4f}, D_loss : {:.4f}".format(loss_g, loss_d))	
		print("  Gs_loss: {:.4f}, Ss_loss: {:.4f}".format(loss_gs, loss_ss))
		print()
		
		x_samp = samp_fig(sess, (6,6), x1_train, x2_train)
		plot_x(i,'samp', x_samp, (6,6))
		i+=1