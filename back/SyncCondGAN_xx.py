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

#==================== Data Batch ====================
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

#==================== Parameter ====================
batch_size = 64

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

#==================== Placeholder ====================
x1_ = tf.placeholder(tf.float32, shape=[None, 784])
x2_ = tf.placeholder(tf.float32, shape=[None, 784])

s_ = tf.placeholder(tf.float32, shape=[None, 1])

#==================== Generator ====================
#Generator
W_g1 = tf.Variable(xavier_init([784, 128]))
b_g1 = tf.Variable(tf.zeros(shape=[128]))

W_g2 = tf.Variable(xavier_init([128, 256]))
b_g2 = tf.Variable(tf.zeros(shape=[256]))

W_g3 = tf.Variable(xavier_init([256,784]))
b_g3 = tf.Variable(tf.zeros(shape=[784]))

var_g = [W_g1, b_g1, W_g2, b_g2, W_g3, b_g3]

def Generator(x):
	h_g1 = tf.nn.relu(tf.matmul(x, W_g1) + b_g1)
	h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
	x_logit = tf.matmul(h_g2, W_g3) + b_g3
	x_prob = tf.nn.sigmoid(x_logit)
	return x_prob

#==================== Discriminator ====================
#Discriminator
W_d1 = tf.Variable(xavier_init([784,128]))
b_d1 = tf.Variable(tf.zeros(shape=[128]))

W_d2 = tf.Variable(xavier_init([128,1]))
b_d2 = tf.Variable(tf.zeros(shape=[1]))

var_d = [W_d1, b_d1, W_d2, b_d2]

def Discriminator(x):
	h_d1 = tf.nn.relu(tf.matmul(x, W_d1) + b_d1)
	y_r_logit = tf.matmul(h_d1, W_d2) + b_d2
	y_r_prob = tf.nn.sigmoid(y_r_logit)
	return y_r_logit, y_r_prob

#==================== Synchronizer ====================
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

G_sample = Generator(x1_)

D_real_logit, D_real_prob = Discriminator(x2_)
D_fake_logit, D_fake_prob = Discriminator(G_sample)

S_real_logit, S_real_prob = Synchronizer(x1_, x2_)
S_fake_logit, S_fake_prob = Synchronizer(x1_, G_sample)

#==================== Loss & Train ====================
#Vanilla GAN Loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logit, labels=tf.ones_like(D_real_logit)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)))
D_loss = D_loss_real + D_loss_fake 

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)))
'''
eps = 1e-8
D_loss = -tf.reduce_mean(tf.log(D_real_prob + eps) + tf.log(1. - D_fake_prob + eps))
G_loss = -tf.reduce_mean(tf.log(D_fake_prob + eps))
'''

#Synchronizer Loss
Ss_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_real_logit, labels=s_))
Ss_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=tf.zeros_like(S_fake_logit)))
Ss_loss = Ss_real_loss + Ss_fake_loss
Gs_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=s_))

#Solver 
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=var_g)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=var_d)

Ss_solver = tf.train.AdamOptimizer().minimize(Ss_loss, var_list=var_s)
Gs_solver = tf.train.AdamOptimizer().minimize(Gs_loss, var_list=var_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#==================== Dataset ====================
mnist_digit = input_data.read_data_sets('MNIST_digit', one_hot=False)
x_digit = mnist_digit.train.images
y_digit = mnist_digit.train.labels
x1_train = class_list(x_digit, y_digit, 10)
'''
mnist_fashion = input_data.read_data_sets('MNIST_fashion', one_hot=False)
x_fashion = mnist_fashion.train.images
y_fashion = mnist_fashion.train.labels
x2_train = class_list(x_fashion, y_fashion, 10)
'''
#Rotatate digit (cross domain)
x_digit_rot = scipy.ndimage.interpolation.rotate(x_digit.reshape(-1, 28, 28), 90, axes=(1, 2)).reshape(-1, 28*28)
x2_train = class_list(x_digit_rot, y_digit, 10)

#==================== Main ====================
if not os.path.exists('out/'):
    os.makedirs('out/')

i=0
for it in range(200001):
	#Get batch training data
	x1_sync, x2_sync, s_sync = sync_match_next_batch(x1_train, x2_train, batch_size)
	x1_nsync, x2_nsync, s_nsync = nsync_match_next_batch(x1_train, x2_train, batch_size)
	
	x1_batch = np.concatenate((x1_sync, x1_nsync), axis=0)
	x2_batch = np.concatenate((x2_sync, x2_nsync), axis=0)
	sr_batch = np.concatenate((s_sync, s_nsync), axis=0)
	sf_batch = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=0)

	#Training
	_, loss_d = sess.run([D_solver, D_loss], feed_dict={x1_:x1_batch, x2_: x2_batch})
	_, loss_ss = sess.run([Ss_solver, Ss_loss], feed_dict={x1_:x1_batch, x2_: x2_batch, s_:sr_batch})
	#loss_ss = 0.

	_, loss_g = sess.run([G_solver, G_loss], feed_dict={x1_:x1_batch})
	_, loss_gs = sess.run([Gs_solver, Gs_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, s_:sf_batch})
	
	#Show result
	if it%1000 == 0:
		print("Iter: {}\n G_loss: {:.4}, Gs_loss: {:.4}\n D_loss: {:.4}, Ss_loss: {:.4}\n"
				.format(it, loss_g, loss_gs, loss_d, loss_ss))

		x1_batch, _, _ = sync_match_next_batch(x1_train, x2_train, 8)
		x2_samp = sess.run(G_sample, feed_dict={x1_: x1_batch})
		x_samp = np.concatenate((x1_batch, x2_samp), axis=0)
		
		plot_x(i,'samp', x_samp)
		i += 1