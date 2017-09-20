import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import math

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def plot_x(id, type, samp):
    fig = plot(samp)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def class_list(imgs, labels, c=10):
	imgs_list = imgs.tolist()
	imgs_class_list = []
	for i in range(c):
		imgs_class_list.append([])
	
	for i in range(labels.shape[0]):
		imgs_class_list[labels[i]].append(imgs_list[i])

	return imgs_class_list

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

def sync_match_next_batch(img1_list, img2_list, size):
	img1_samp = []
	img2_samp = []

	for i in range(size):
		n = random.randint(0, len(img1_list)-1)
		r = random.randint(0, 2000)

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

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def sample_normal_z(m, n):
	return np.random.normal(0., 1., size=[m, n])

#Parameter
batch_size = 128
z_dim = 31
c_dim = 1

#Model Build
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

#Placeholder
z1_ = tf.placeholder(tf.float32, shape=[None, z_dim])
z2_ = tf.placeholder(tf.float32, shape=[None, z_dim])

c1_ = tf.placeholder(tf.float32, shape=[None, c_dim])
c2_ = tf.placeholder(tf.float32, shape=[None, c_dim])

x1_ = tf.placeholder(tf.float32, shape=[None, 784])
x2_ = tf.placeholder(tf.float32, shape=[None, 784])

s_ = tf.placeholder(tf.float32, shape=[None, 1])

#Generator
W_s_g_fc1 = tf.Variable(xavier_init([z_dim+c_dim,7*7*32]))
b_s_g_fc1 = tf.Variable(tf.zeros(shape=[7*7*32]))

W_m1_g_conv2 = tf.Variable(xavier_init([5,5,16,32]))
b_m1_g_conv2 = tf.Variable(tf.zeros(shape=[16]))
W_m2_g_conv2 = tf.Variable(xavier_init([5,5,16,32]))
b_m2_g_conv2 = tf.Variable(tf.zeros(shape=[16]))
W_m_g_conv2 = [W_m1_g_conv2, W_m2_g_conv2]
b_m_g_conv2 = [b_m1_g_conv2, b_m2_g_conv2]

W_m1_g_conv3 = tf.Variable(xavier_init([5,5,1,16]))
b_m1_g_conv3 = tf.Variable(tf.zeros(shape=[1]))
W_m2_g_conv3 = tf.Variable(xavier_init([5,5,1,16]))
b_m2_g_conv3 = tf.Variable(tf.zeros(shape=[1]))
W_m_g_conv3 = [W_m1_g_conv3, W_m2_g_conv3]
b_m_g_conv3 = [b_m1_g_conv3, b_m2_g_conv3]

var_gs = [W_s_g_fc1, b_s_g_fc1]
var_g1 = [W_m1_g_conv2, b_m1_g_conv2, W_m1_g_conv3, b_m1_g_conv3]
var_g2 = [W_m2_g_conv2, b_m2_g_conv2, W_m2_g_conv3, b_m2_g_conv3]

def Generator(z, c, m):
	z_c = tf.concat(axis=1, values=[z, c])
	h_g_fc1 = tf.nn.relu(tf.matmul(z_c, W_s_g_fc1) + b_s_g_fc1)
	h_g_re1 = tf.reshape(h_g_fc1, [-1, 7, 7, 32])

	output_shape_g2 = tf.stack([tf.shape(z)[0], 14, 14, 16])
	h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_m_g_conv2[m], output_shape_g2) + b_m_g_conv2[m])

	output_shape_g3 = tf.stack([tf.shape(z)[0], 28, 28, 1])
	h_g_conv3 = tf.nn.sigmoid(deconv2d(h_g_conv2, W_m_g_conv3[m], output_shape_g3) + b_m_g_conv3[m])

	h_g_re3 = tf.reshape(h_g_conv3, [-1,784])
	return h_g_re3

#Discriminator
W_m1_d_conv1 = tf.Variable(xavier_init([5,5,1,4]))
b_m1_d_conv1 = tf.Variable(tf.zeros(shape=[4]))
W_m2_d_conv1 = tf.Variable(xavier_init([5,5,1,4]))
b_m2_d_conv1 = tf.Variable(tf.zeros(shape=[4]))
W_m_d_conv1 = [W_m1_d_conv1, W_m2_d_conv1]
b_m_d_conv1 = [b_m1_d_conv1, b_m2_d_conv1]

W_m1_d_conv2 = tf.Variable(xavier_init([5,5,4,8]))
b_m1_d_conv2 = tf.Variable(tf.zeros(shape=[8]))
W_m2_d_conv2 = tf.Variable(xavier_init([5,5,4,8]))
b_m2_d_conv2 = tf.Variable(tf.zeros(shape=[8]))
W_m_d_conv2 = [W_m1_d_conv2, W_m2_d_conv2]
b_m_d_conv2 = [b_m1_d_conv2, b_m2_d_conv2]

W_m1_d_fc3 = tf.Variable(xavier_init([7*7*8, 64]))
b_m1_d_fc3 = tf.Variable(tf.zeros(shape=[64]))
W_m2_d_fc3 = tf.Variable(xavier_init([7*7*8, 64]))
b_m2_d_fc3 = tf.Variable(tf.zeros(shape=[64]))
W_m_d_fc3 = [W_m1_d_fc3, W_m2_d_fc3]
b_m_d_fc3 = [b_m1_d_fc3, b_m2_d_fc3]

W_m1_d_fc4 = tf.Variable(xavier_init([64, 1]))
b_m1_d_fc4 = tf.Variable(tf.zeros(shape=[1]))
W_m2_d_fc4 = tf.Variable(xavier_init([64, 1]))
b_m2_d_fc4 = tf.Variable(tf.zeros(shape=[1]))
W_m_d_fc4 = [W_m1_d_fc4, W_m2_d_fc4]
b_m_d_fc4 = [b_m1_d_fc4, b_m2_d_fc4]

var_d1 = [	W_m1_d_conv1, b_m1_d_conv1,
			W_m1_d_conv2, b_m1_d_conv2,
			W_m1_d_fc3, b_m1_d_fc3,
			W_m1_d_fc4, b_m1_d_fc4	]

var_d2 = [	W_m2_d_conv1, b_m2_d_conv1,
			W_m2_d_conv2, b_m2_d_conv2,
			W_m2_d_fc3, b_m2_d_fc3,
			W_m2_d_fc4, b_m2_d_fc4	]

def Discriminator(x, m):
	x_re = tf.reshape(x, [-1,28,28,1])
	h_d_conv1 = tf.nn.relu(conv2d(x_re, W_m_d_conv1[m], [1,2,2,1]) + b_m_d_conv1[m])

	h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_m_d_conv2[m], [1,2,2,1]) + b_m_d_conv2[m])
	h_d_re2 = tf.reshape(h_d_conv2, [-1,7*7*8])

	h_d_fc3 = tf.nn.relu(tf.matmul(h_d_re2, W_m_d_fc3[m]) + b_m_d_fc3[m])
	
	y_logit = tf.matmul(h_d_fc3, W_m_d_fc4[m]) + b_m_d_fc4[m]
	y_prob = tf.nn.sigmoid(y_logit)
	
	return y_prob, y_logit

#Synchronizer
W_m1_s1 = tf.Variable(xavier_init([784,256]))
b_m1_s1 = tf.Variable(tf.zeros(shape=[256]))
W_m2_s1 = tf.Variable(xavier_init([784,256]))
b_m2_s1 = tf.Variable(tf.zeros(shape=[256]))

W_s_s2 = tf.Variable(xavier_init([512,256]))
b_s_s2 = tf.Variable(tf.zeros(shape=[256]))

W_s_s3 = tf.Variable(xavier_init([256,1]))
b_s_s3 = tf.Variable(tf.zeros(shape=[1]))

var_s = [ W_m1_s1, b_m1_s1, 
		  W_m2_s1, b_m2_s1,
		  W_s_s2, b_s_s2,
		  W_s_s3, b_s_s3 ]

def Synchronizer(x1, x2):
	h_m1_s1 = tf.nn.relu(tf.matmul(x1, W_m1_s1) + b_m1_s1)
	h_m2_s1 = tf.nn.relu(tf.matmul(x2, W_m2_s1) + b_m2_s1)

	h_concat_s1 = tf.concat(axis=1, values=[h_m1_s1, h_m2_s1])
	h_s2 = tf.nn.relu(tf.matmul(h_concat_s1, W_s_s2) + b_s_s2)
	y_s_logit = tf.matmul(h_s2, W_s_s3) + b_s_s3
	y_s_prob = tf.nn.sigmoid(y_s_logit)
	return y_s_logit, y_s_prob

G1_sample = Generator(z1_, c1_, 0)
G2_sample = Generator(z2_, c2_, 1)

D1_real_logit, D1_real_prob = Discriminator(x1_, 0)
D2_real_logit, D2_real_prob = Discriminator(x2_, 1)
D1_fake_logit, D1_fake_prob = Discriminator(G1_sample, 0)
D2_fake_logit, D2_fake_prob = Discriminator(G2_sample, 1)

S_real_logit, S_real_prob = Synchronizer(x1_, x2_)
S_fake_logit, S_fake_prob = Synchronizer(G1_sample, G2_sample)

#Train Generator & Discriminator

#Vanilla GAN Loss
D1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_real_logit, labels=tf.ones_like(D1_real_logit)))
D1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake_logit, labels=tf.zeros_like(D1_fake_logit)))
D2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_real_logit, labels=tf.ones_like(D2_real_logit)))
D2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake_logit, labels=tf.zeros_like(D2_fake_logit)))
D1_loss = D1_loss_real + D1_loss_fake 
D2_loss = D2_loss_real + D2_loss_fake

G1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake_logit, labels=tf.ones_like(D1_fake_logit)))
G2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake_logit, labels=tf.ones_like(D2_fake_logit)))
'''
#W-GAN Loss
eps = 1e-8
D1_loss = -tf.reduce_mean(tf.log(D1_real_prob + eps) + tf.log(1. - D1_fake_prob + eps))
D2_loss = -tf.reduce_mean(tf.log(D2_real_prob + eps) + tf.log(1. - D2_fake_prob + eps))

G1_loss = -tf.reduce_mean(tf.log(D1_fake_prob + eps))
G2_loss = -tf.reduce_mean(tf.log(D2_fake_prob + eps))
'''

#Train Synchronizer
S_real_loss = tf.reduce_mean(tf.reduce_sum(tf.square(S_real_prob - s_), reduction_indices=[1]))
S_fake_loss = tf.reduce_mean(tf.reduce_sum(tf.square(S_fake_prob - s_), reduction_indices=[1]))

#Solver 
G1_solver = tf.train.AdamOptimizer().minimize(G1_loss, var_list=var_gs + var_g1)
G2_solver = tf.train.AdamOptimizer().minimize(G2_loss, var_list=var_gs + var_g2)

D1_solver = tf.train.AdamOptimizer().minimize(D1_loss, var_list=var_d1)
D2_solver = tf.train.AdamOptimizer().minimize(D2_loss, var_list=var_d2)

S_real_solver = tf.train.AdamOptimizer().minimize(S_real_loss, var_list=var_s)
S_fake_solver = tf.train.AdamOptimizer().minimize(S_fake_loss, var_list=var_gs + var_g1 + var_g2)

#Read Dataset
mnist_digit = input_data.read_data_sets('MNIST_digit', one_hot=False)
x_digit = mnist_digit.train.images
y_digit = mnist_digit.train.labels
x2_train = class_list(x_digit, y_digit, 10)

mnist_fashion = input_data.read_data_sets('MNIST_fashion', one_hot=False)
x_fashion = mnist_fashion.train.images
y_fashion = mnist_fashion.train.labels
x1_train = class_list(x_fashion, y_fashion, 10)
'''
#Rotatate digit (cross domain)
x_digit_rot = scipy.ndimage.interpolation.rotate(x_digit.reshape(-1, 28, 28), 90, axes=(1, 2)).reshape(-1, 28*28)
x2_train = class_list(x_digit_rot, y_digit, 10)
'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Main
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(500001):
	#Get batch training data
	x1_sync, x2_sync, s_sync = sync_match_next_batch(x1_train, x2_train, batch_size)
	x1_nsync, x2_nsync, s_nsync = nsync_match_next_batch(x1_train, x2_train, batch_size)
	
	x1_batch = np.concatenate((x1_sync, x1_nsync), axis=0)
	x2_batch = np.concatenate((x2_sync, x2_nsync), axis=0)
	sr_batch = np.concatenate((s_sync, s_nsync), axis=0)

	z1_batch = sample_z(batch_size*2, z_dim)
	z2_batch = sample_z(batch_size*2, z_dim)

	c_sync_batch = sample_z(batch_size, c_dim)
	c1_nsync_batch = sample_z(batch_size, c_dim)
	c2_nsync_batch = sample_z(batch_size, c_dim)

	c1_batch = np.concatenate((c_sync_batch, c1_nsync_batch), axis=0)
	c2_batch = np.concatenate((c_sync_batch, c2_nsync_batch), axis=0)
	sf_batch = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=0)

	x1_batch, _= mnist_digit.train.next_batch(batch_size*2)

	#Training
	_, loss_d1 = sess.run([D1_solver, D1_loss], feed_dict={z1_:z1_batch, c1_:c1_batch, x1_:x1_batch})
	#_, loss_d2 = sess.run([D2_solver, D2_loss], feed_dict={z2_:z2_batch, c2_:c2_batch, x2_:x2_batch})

	_, loss_g1 = sess.run([G1_solver, G1_loss], feed_dict={z1_:z1_batch, c1_:c1_batch})
	#_, loss_g2 = sess.run([G2_solver, G2_loss], feed_dict={z2_:z2_batch, c2_:c2_batch})

	#_, loss_sr = sess.run([S_real_solver, S_real_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, s_:sr_batch})
	#_, loss_sf = sess.run([S_fake_solver, S_fake_loss], feed_dict={z1_:z1_batch, z2_:z2_batch, c1_:c1_batch, c2_:c2_batch, s_:sf_batch})
	
	#Show result
	if it%100 == 0:
		#print("Iter: {}\n D1_loss: {:.4}, D2_loss: {:.4}\n G1_loss: {:.4}, G2_loss: {:.4}\n Sr_loss: {:.4}, Sf_loss: {:.4}\n"
		#		.format(it, loss_d1, loss_d2, loss_g1, loss_g2, loss_sr, loss_sf))
		print("Iter: {}\n G1_loss: {:.4}, D1_loss: {:.4}".format(it, loss_g1, loss_d1))	
		z1_batch = sample_z(8, z_dim)
		z2_batch = sample_z(8, z_dim)
		c_batch = sample_z(8, c_dim)
		
		x1_samp = sess.run(G1_sample, feed_dict={z1_: z1_batch, c1_: c_batch})
		x2_samp = sess.run(G2_sample, feed_dict={z2_: z2_batch, c2_: c_batch})
		x_samp = np.concatenate((x1_samp, x2_samp), axis=0)
		
		plot_x(i,'samp', x_samp)
		i += 1