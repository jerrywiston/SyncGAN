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

def next_batch(imgs, size):
    img_samp = np.ndarray(shape=(size, imgs.shape[1]))
    for i in range(size):
        r = random.randint(0,imgs.shape[0]-1)
        img_samp[i] = imgs[r]
    return img_samp

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def xavier_init(size):
    if len(size) == 4:
        n_inputs = size[0]*size[1]*size[2]
        n_outputs = size[3]
    else:
        n_inputs = size[0]
        n_outputs = size[1]
    
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal(size, stddev=stddev)

#Parameter
z_dim = 32
batch_size = 128

#Placeholder
z_ = tf.placeholder(tf.float32, shape=[None, z_dim])
x_ = tf.placeholder(tf.float32, shape=[None, 784])

#Generator
W_g_fc1 = tf.Variable(xavier_init([z_dim,7*7*32]))
b_g_fc1 = tf.Variable(tf.zeros(shape=[7*7*32]))

W_g_conv2 = tf.Variable(xavier_init([5,5,16,32]))
b_g_conv2 = tf.Variable(tf.zeros(shape=[16]))

W_g_conv3 = tf.Variable(xavier_init([5,5,1,16]))
b_g_conv3 = tf.Variable(tf.zeros(shape=[1]))

var_g = [W_g_fc1, b_g_fc1, W_g_conv2, b_g_conv2, W_g_conv3, b_g_conv3]

def conv2d(x, W, stride):
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride=[1,2,2,1]):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

def Generator(z):
    h_g_fc1 = tf.nn.relu(tf.matmul(z, W_g_fc1) + b_g_fc1)
    h_g_re1 = tf.reshape(h_g_fc1, [-1, 7, 7, 32])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 14, 14, 16])
    h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_g_conv2, output_shape_g2) + b_g_conv2)

    output_shape_g3 = tf.stack([tf.shape(z)[0], 28, 28, 1])
    h_g_conv3 = tf.nn.sigmoid(deconv2d(h_g_conv2, W_g_conv3, output_shape_g3) + b_g_conv3)

    h_g_re3 = tf.reshape(h_g_conv3, [-1,784])
    return h_g_re3

#Discriminator
W_d_conv1 = tf.Variable(xavier_init([5,5,1,4]))
b_d_conv1 = tf.Variable(tf.zeros(shape=[4]))

W_d_conv2 = tf.Variable(xavier_init([5,5,4,8]))
b_d_conv2 = tf.Variable(tf.zeros(shape=[8]))

W_d_fc3 = tf.Variable(xavier_init([7*7*8, 64]))
b_d_fc3 = tf.Variable(tf.zeros(shape=[64]))

W_d_fc4 = tf.Variable(xavier_init([64, 1]))
b_d_fc4 = tf.Variable(tf.zeros(shape=[1]))

var_d = [W_d_conv1, b_d_conv1, W_d_conv2, b_d_conv2, W_d_fc3, b_d_fc3, W_d_fc4, b_d_fc4]

def Discriminator(x):
	x_re = tf.reshape(x, [-1,28,28,1])
	h_d_conv1 = tf.nn.relu(conv2d(x_re, W_d_conv1, [1,2,2,1]) + b_d_conv1)

	h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_d_conv2, [1,2,2,1]) + b_d_conv2)
	h_d_re2 = tf.reshape(h_d_conv2, [-1,7*7*8])

	h_d_fc3 = tf.nn.relu(tf.matmul(h_d_re2, W_d_fc3) + b_d_fc3)
	
	y_logit = tf.matmul(h_d_fc3, W_d_fc4) + b_d_fc4
	y_prob = tf.nn.sigmoid(y_logit)
	
	return y_prob, y_logit 

G_sample = Generator(z_)
D_real, D_logit_real = Discriminator(x_)
D_fake, D_logit_fake = Discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=var_d)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=var_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Main
if not os.path.exists('out/'):
    os.makedirs('out/')

mnist = input_data.read_data_sets('MNIST_digit', one_hot=True)
#mnist = input_data.read_data_sets('MNIST_fashion', one_hot=True)

i=0
for it in range(20001):
    #Train weight & latent
    x_batch, _= mnist.train.next_batch(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x_: x_batch, z_: sample_z(batch_size, z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z_: sample_z(batch_size, z_dim)})

    #Show result
    if it % 100 == 0:
        print('Iter: {}, G_loss: {:.4}, D_loss: {:.4}'.format(it, G_loss_curr, D_loss_curr))
        z_samp = sample_z(16, z_dim)
        x_samp = sess.run(G_sample, feed_dict={z_: z_samp})
        plot_x(i,'samp', x_samp)
        i += 1
