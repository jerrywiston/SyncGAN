import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import math
import scipy.ndimage.interpolation
from PIL import Image

#==================== Draw Figure ====================
def plot(x1_samp, x2_samp, size):
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(size[0], size[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(size[0]):
        for j in range(size[1]):
            ax = plt.subplot(gs[i * size[0] + j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if i % 2 == 0:
                plt.imshow(x1_samp[int(i / 2)][j].reshape(32, 32), cmap='Greys_r')
            else:
                plt.imshow(x2_samp[int(i / 2)][j])
        

    return fig

def plot_x(id, x1_samp, x2_samp, size=(4,4)):
    fig = plot(x1_samp, x2_samp, size)
    plt.savefig('out/{}.png'.format(str(id).zfill(4)), bbox_inches='tight')
    plt.close(fig)

def samp_fig(sess, size):
    x1_samp = np.zeros([int(size[0] / 2), size[1], 32, 32, 1], dtype=np.float32)
    x2_samp = np.zeros([int(size[0] / 2), size[1], 32, 32, 3], dtype=np.float32)

    for i in range(int(size[0]/2)):
        z_sync_batch = sample_z(size[1], z_dim)
        x1_samp[i] = sess.run(G1_sample, feed_dict={z1_: z_sync_batch})
        x2_samp[i] = sess.run(G2_sample, feed_dict={z2_: z_sync_batch})

    return x1_samp, x2_samp

#============================== Read Data ==============================

def read_data(size):
    sketch_list = np.zeros([size, 32, 32, 1], dtype=np.float32)
    celebA_list = np.zeros([size, 32, 32, 3], dtype=np.float32)
    
    for i in range(size):
        fname = str(i).zfill(6) + '.jpg'    
        sketch_list[i] = np.array(Image.open(sketch_dir + fname).convert('L')).reshape((32, 32, 1)) / 255.
        celebA_list[i] = np.array(Image.open(celebA_dir + fname)) / 255.
        
    return sketch_list, celebA_list


#============================== Batch ==============================

def sync_next_batch(list_1, list_2, size):
    batch_1 = np.zeros([size, list_1.shape[1], list_1.shape[2], list_1.shape[3]])
    batch_2 = np.zeros([size, list_2.shape[1], list_2.shape[2], list_2.shape[3]])

    for i in range(size):
        n = random.randint(0, list_1.shape[0] - 1)
        batch_1[i] = list_1[n]
        batch_2[i] = list_2[n]

    return batch_1, batch_2, np.ones((size, 1))

def nsync_next_batch(list_1, list_2, size):
    batch_1 = np.zeros([size, list_1.shape[1], list_1.shape[2], list_1.shape[3]])
    batch_2 = np.zeros([size, list_2.shape[1], list_2.shape[2], list_2.shape[3]])

    for i in range(size):
        n1 = random.randint(0, list_1.shape[0] - 1)
        n2 = random.randint(0, list_2.shape[0] - 1)
        while n1 == n2:
            n2 = random.randint(0, list_2.shape[0] - 1)

        batch_1[i] = list_1[n1]
        batch_2[i] = list_2[n2]

    return batch_1, batch_2, np.zeros((size, 1))

def sample_z(m, n, type=1):
    if type == 0:
    	return np.random.uniform(-1., 1., size=[m, n])
    else:
    	return np.random.normal(0., 1., size=[m, n])

#==================== Parameter ====================
batch_size = 128
z_dim = 64

def xavier_init(size):
    if len(size) == 4:
        n_inputs = size[0]*size[1]*size[2]
        n_outputs = size[3]
    else:
        n_inputs = size[0]
        n_outputs = size[1]
    
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal(size, stddev=stddev)

def batch_normalization(x, is_training=True):
    return tf.contrib.layers.batch_norm(
            x,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            is_training=is_training
    )

def conv2d(x, W, stride, bn=True):
    if bn:
        x = batch_normalization(x, is_training=True)
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride=[1,2,2,1], bn=True):
    if bn:
        x = batch_normalization(x, is_training=True)
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

#==================== Placeholder ====================
z1_ = tf.placeholder(tf.float32, shape=[None, z_dim])
z2_ = tf.placeholder(tf.float32, shape=[None, z_dim])

x1_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
x2_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

s_ = tf.placeholder(tf.float32, shape=[None, 1])

#==================== Generator ====================
#Generator 1
W_m1_g_fc1 = tf.Variable(xavier_init([z_dim,2*2*1024]))
b_m1_g_fc1 = tf.Variable(tf.zeros(shape=[2*2*1024]))

W_m1_g_conv2 = tf.Variable(xavier_init([5,5,512,1024]))
b_m1_g_conv2 = tf.Variable(tf.zeros(shape=[512]))

W_m1_g_conv3 = tf.Variable(xavier_init([5,5,256,512]))
b_m1_g_conv3 = tf.Variable(tf.zeros(shape=[256]))

W_m1_g_conv4 = tf.Variable(xavier_init([5,5,128,256]))
b_m1_g_conv4 = tf.Variable(tf.zeros(shape=[128]))

W_m1_g_conv5 = tf.Variable(xavier_init([5,5,64,128]))
b_m1_g_conv5 = tf.Variable(tf.zeros(shape=[64]))

W_m1_g_conv6 = tf.Variable(xavier_init([3,3,1,64]))
b_m1_g_conv6 = tf.Variable(tf.zeros(shape=[1]))

var_g1 = [W_m1_g_fc1, b_m1_g_fc1, 
         W_m1_g_conv2, b_m1_g_conv2, 
         W_m1_g_conv3, b_m1_g_conv3, 
         W_m1_g_conv4, b_m1_g_conv4, 
         W_m1_g_conv5, b_m1_g_conv5, 
         W_m1_g_conv6, b_m1_g_conv6]

def Generator1(z):
    h_g_fc1 = tf.nn.relu(tf.matmul(z, W_m1_g_fc1) + b_m1_g_fc1)
    h_g_re1 = tf.reshape(h_g_fc1, [-1, 2, 2, 1024])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 4, 4, 512])
    h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_m1_g_conv2, output_shape_g2) + b_m1_g_conv2)

    output_shape_g3 = tf.stack([tf.shape(z)[0], 8, 8, 256])
    h_g_conv3 = tf.nn.relu(deconv2d(h_g_conv2, W_m1_g_conv3, output_shape_g3) + b_m1_g_conv3)

    output_shape_g4 = tf.stack([tf.shape(z)[0], 16, 16, 128])
    h_g_conv4 = tf.nn.relu(deconv2d(h_g_conv3, W_m1_g_conv4, output_shape_g4) + b_m1_g_conv4)

    output_shape_g5 = tf.stack([tf.shape(z)[0], 32, 32, 64])
    h_g_conv5 = tf.nn.relu(deconv2d(h_g_conv4, W_m1_g_conv5, output_shape_g5) + b_m1_g_conv5)

    output_shape_g6 = tf.stack([tf.shape(z)[0], 32, 32, 1])
    h_g_conv6 = tf.nn.sigmoid(deconv2d(h_g_conv5, W_m1_g_conv6, output_shape_g6, stride=[1,1,1,1]) + b_m1_g_conv6)

    return h_g_conv6

#Generator 2
W_m2_g_fc1 = tf.Variable(xavier_init([z_dim,2*2*1024]))
b_m2_g_fc1 = tf.Variable(tf.zeros(shape=[2*2*1024]))

W_m2_g_conv2 = tf.Variable(xavier_init([5,5,512,1024]))
b_m2_g_conv2 = tf.Variable(tf.zeros(shape=[512]))

W_m2_g_conv3 = tf.Variable(xavier_init([5,5,256,512]))
b_m2_g_conv3 = tf.Variable(tf.zeros(shape=[256]))

W_m2_g_conv4 = tf.Variable(xavier_init([5,5,128,256]))
b_m2_g_conv4 = tf.Variable(tf.zeros(shape=[128]))

W_m2_g_conv5 = tf.Variable(xavier_init([5,5,64,128]))
b_m2_g_conv5 = tf.Variable(tf.zeros(shape=[64]))

W_m2_g_conv6 = tf.Variable(xavier_init([3,3,3,64]))
b_m2_g_conv6 = tf.Variable(tf.zeros(shape=[3]))

var_g2 = [W_m2_g_fc1, b_m2_g_fc1, 
         W_m2_g_conv2, b_m2_g_conv2, 
         W_m2_g_conv3, b_m2_g_conv3, 
         W_m2_g_conv4, b_m2_g_conv4, 
         W_m2_g_conv5, b_m2_g_conv5, 
         W_m2_g_conv6, b_m2_g_conv6]

def Generator2(z):
    h_g_fc1 = tf.nn.relu(tf.matmul(z, W_m2_g_fc1) + b_m2_g_fc1)
    h_g_re1 = tf.reshape(h_g_fc1, [-1, 2, 2, 1024])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 4, 4, 512])
    h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_m2_g_conv2, output_shape_g2) + b_m2_g_conv2)

    output_shape_g3 = tf.stack([tf.shape(z)[0], 8, 8, 256])
    h_g_conv3 = tf.nn.relu(deconv2d(h_g_conv2, W_m2_g_conv3, output_shape_g3) + b_m2_g_conv3)

    output_shape_g4 = tf.stack([tf.shape(z)[0], 16, 16, 128])
    h_g_conv4 = tf.nn.relu(deconv2d(h_g_conv3, W_m2_g_conv4, output_shape_g4) + b_m2_g_conv4)

    output_shape_g5 = tf.stack([tf.shape(z)[0], 32, 32, 64])
    h_g_conv5 = tf.nn.relu(deconv2d(h_g_conv4, W_m2_g_conv5, output_shape_g5) + b_m2_g_conv5)

    output_shape_g6 = tf.stack([tf.shape(z)[0], 32, 32, 3])
    h_g_conv6 = tf.nn.sigmoid(deconv2d(h_g_conv5, W_m2_g_conv6, output_shape_g6, stride=[1,1,1,1]) + b_m2_g_conv6)

    return h_g_conv6

#==================== Discriminator ====================
#Discriminator 1
W_m1_d_conv1 = tf.Variable(xavier_init([5,5,1,32]))
b_m1_d_conv1 = tf.Variable(tf.zeros(shape=[32]))

W_m1_d_conv2 = tf.Variable(xavier_init([5,5,32,64]))
b_m1_d_conv2 = tf.Variable(tf.zeros(shape=[64]))

W_m1_d_conv3 = tf.Variable(xavier_init([5,5,64,128]))
b_m1_d_conv3 = tf.Variable(tf.zeros(shape=[128]))

W_m1_d_conv4 = tf.Variable(xavier_init([5,5,128,256]))
b_m1_d_conv4 = tf.Variable(tf.zeros(shape=[256]))

W_m1_d_fc5 = tf.Variable(xavier_init([256,1]))
b_m1_d_fc5 = tf.Variable(tf.zeros(shape=[1]))

var_d1 = [W_m1_d_conv1, b_m1_d_conv1, 
          W_m1_d_conv2, b_m1_d_conv2, 
          W_m1_d_conv3, b_m1_d_conv3, 
          W_m1_d_conv4, b_m1_d_conv4, 
          W_m1_d_fc5, b_m1_d_fc5]

def Discriminator1(x):
    h_d_conv1 = tf.nn.relu(conv2d(x, W_m1_d_conv1, [1,2,2,1], bn=False) + b_m1_d_conv1)
    h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_m1_d_conv2, [1,2,2,1]) + b_m1_d_conv2)
    h_d_conv3 = tf.nn.relu(conv2d(h_d_conv2, W_m1_d_conv3, [1,2,2,1]) + b_m1_d_conv3)
    h_d_conv4 = tf.nn.relu(conv2d(h_d_conv3, W_m1_d_conv4, [1,2,2,1]) + b_m1_d_conv4)

    avg_pool = tf.reduce_mean(h_d_conv4, [1, 2])

    y_logit = tf.matmul(avg_pool, W_m1_d_fc5) + b_m1_d_fc5
    y_prob = tf.nn.sigmoid(y_logit)

    return y_prob, y_logit

#Discriminator 2
W_m2_d_conv1 = tf.Variable(xavier_init([5,5,3,32]))
b_m2_d_conv1 = tf.Variable(tf.zeros(shape=[32]))

W_m2_d_conv2 = tf.Variable(xavier_init([5,5,32,64]))
b_m2_d_conv2 = tf.Variable(tf.zeros(shape=[64]))

W_m2_d_conv3 = tf.Variable(xavier_init([5,5,64,128]))
b_m2_d_conv3 = tf.Variable(tf.zeros(shape=[128]))

W_m2_d_conv4 = tf.Variable(xavier_init([5,5,128,256]))
b_m2_d_conv4 = tf.Variable(tf.zeros(shape=[256]))

W_m2_d_fc5 = tf.Variable(xavier_init([256,1]))
b_m2_d_fc5 = tf.Variable(tf.zeros(shape=[1]))

var_d2 = [W_m2_d_conv1, b_m2_d_conv1, 
          W_m2_d_conv2, b_m2_d_conv2, 
          W_m2_d_conv3, b_m2_d_conv3, 
          W_m2_d_conv4, b_m2_d_conv4, 
          W_m2_d_fc5, b_m2_d_fc5]

def Discriminator2(x):
    h_d_conv1 = tf.nn.relu(conv2d(x, W_m2_d_conv1, [1,2,2,1], bn=False) + b_m2_d_conv1)
    h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_m2_d_conv2, [1,2,2,1]) + b_m2_d_conv2)
    h_d_conv3 = tf.nn.relu(conv2d(h_d_conv2, W_m2_d_conv3, [1,2,2,1]) + b_m2_d_conv3)
    h_d_conv4 = tf.nn.relu(conv2d(h_d_conv3, W_m2_d_conv4, [1,2,2,1]) + b_m2_d_conv4)

    avg_pool = tf.reduce_mean(h_d_conv4, [1, 2])

    y_logit = tf.matmul(avg_pool, W_m2_d_fc5) + b_m2_d_fc5
    y_prob = tf.nn.sigmoid(y_logit)

    return y_prob, y_logit

#==================== Synchronizer ====================
#Mode 1
W_m1_s_conv1 = tf.Variable(xavier_init([5,5,1,16]))
b_m1_s_conv1 = tf.Variable(tf.zeros(shape=[16]))
W_m1_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m1_s_conv2 = tf.Variable(tf.zeros(shape=[32]))
W_m1_s_fc3 = tf.Variable(xavier_init([8*8*32, 256]))
b_m1_s_fc3 = tf.Variable(tf.zeros(shape=[256]))

#Modal 2
W_m2_s_conv1 = tf.Variable(xavier_init([5,5,3,16]))
b_m2_s_conv1 = tf.Variable(tf.zeros(shape=[16]))
W_m2_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_s_conv2 = tf.Variable(tf.zeros(shape=[32]))
W_m2_s_fc3 = tf.Variable(xavier_init([8*8*32, 256]))
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
	h_conv1 = tf.nn.relu(conv2d(x1, W_m1_s_conv1, [1,2,2,1], bn=False) + b_m1_s_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_m1_s_conv2, [1,2,2,1]) + b_m1_s_conv2)
	h_re2 = tf.reshape(h_conv2, [-1,8*8*32])
	v1 = tf.nn.relu(tf.matmul(h_re2, W_m1_s_fc3) + b_m1_s_fc3)

	#Modal 2
	h_m2_s_conv1 = tf.nn.relu(conv2d(x2, W_m2_s_conv1, [1,2,2,1], bn=False) + b_m2_s_conv1)
	h_m2_s_conv2 = tf.nn.relu(conv2d(h_m2_s_conv1, W_m2_s_conv2, [1,2,2,1]) + b_m2_s_conv2)
	h_m2_s_re2 = tf.reshape(h_m2_s_conv2, [-1,8*8*32])
	v2 = tf.nn.relu(tf.matmul(h_m2_s_re2, W_m2_s_fc3) + b_m2_s_fc3)

	#Shared
	v = tf.concat(axis=1, values=[v1, v2])
	h_s4 = tf.nn.relu(tf.matmul(v, W_s_s4) + b_s_s4)
	s_logit = tf.matmul(h_s4, W_s_s5) + b_s_s5
	s_prob = tf.nn.sigmoid(s_logit)
	return s_prob, s_logit

G1_sample = Generator1(z1_)
G2_sample = Generator2(z2_)

D1_real_prob, D1_real_logit = Discriminator1(x1_)
D1_fake_prob, D1_fake_logit = Discriminator1(G1_sample)

D2_real_prob, D2_real_logit = Discriminator2(x2_)
D2_fake_prob, D2_fake_logit = Discriminator2(G2_sample)

S_real_prob, S_real_logit = Synchronizer(x1_, x2_)
S_fake_prob, S_fake_logit = Synchronizer(G1_sample, G2_sample)

#==================== Loss & Train ====================
#Vanilla GAN Loss
D1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_real_logit, labels=tf.ones_like(D1_real_logit)))
D1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake_logit, labels=tf.zeros_like(D1_fake_logit)))
D2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_real_logit, labels=tf.ones_like(D2_real_logit)))
D2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake_logit, labels=tf.zeros_like(D2_fake_logit)))
D1_loss = D1_loss_real + D1_loss_fake 
D2_loss = D2_loss_real + D2_loss_fake

G1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake_logit, labels=tf.ones_like(D1_fake_logit)))
G2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake_logit, labels=tf.ones_like(D2_fake_logit)))

#Synchronize Loss
Ss_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_real_logit, labels=s_))
Ss_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=tf.zeros_like(S_fake_logit)))
Gs_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=s_))

#Solver 
G1_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(G1_loss, var_list=var_g1)
G2_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(G2_loss, var_list=var_g2)

D1_solver = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(D1_loss, var_list=var_d1)
D2_solver = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(D2_loss, var_list=var_d2)

Ss_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(Ss_loss, var_list=var_s)
Gs_solver = tf.train.AdamOptimizer(5e-4, beta1=0.5).minimize(Gs_loss, var_list=var_g1 + var_g2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#==================== cifar handling ====================
sketch_dir = 'img_align_celeba_crop_sketch_32/'
celebA_dir = 'img_align_celeba_crop_32/'

x1_train, x2_train = read_data(10000)
#==================== Main ====================
if not os.path.exists('out/'):
    os.makedirs('out/')

m1_stop = 5000
i=0
for it in range(200001):
	#Get batch training data
	x1_sync, x2_sync, s_sync = sync_next_batch(x1_train, x2_train, batch_size)
	x1_nsync, x2_nsync, s_nsync = nsync_next_batch(x1_train, x2_train, batch_size)
	
	x1_batch = np.concatenate((x1_sync, x1_nsync), axis=0)
	x2_batch = np.concatenate((x2_sync, x2_nsync), axis=0)
	sr_batch = np.concatenate((s_sync, s_nsync), axis=0)

	z_sync_batch = sample_z(batch_size, z_dim)
	z1_nsync_batch = sample_z(batch_size, z_dim)
	z2_nsync_batch = sample_z(batch_size, z_dim)

	z1_batch = np.concatenate((z_sync_batch, z1_nsync_batch), axis=0)
	z2_batch = np.concatenate((z_sync_batch, z2_nsync_batch), axis=0)
	sf_batch = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=0)

	#Training
	_, loss_d1 = sess.run([D1_solver, D1_loss], feed_dict={z1_:z1_batch, x1_:x1_batch})
	_, loss_d2 = sess.run([D2_solver, D2_loss], feed_dict={z2_:z2_batch, x2_:x2_batch})
	_, loss_ss = sess.run([Ss_solver, Ss_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, s_:sr_batch})
	
	_, loss_g1 = sess.run([G1_solver, G1_loss], feed_dict={z1_:z1_batch})
	_, loss_g2 = sess.run([G2_solver, G2_loss], feed_dict={z2_:z2_batch})
	_, loss_gs = sess.run([Gs_solver, Gs_loss], feed_dict={z1_:z1_batch, z2_:z2_batch, s_:sf_batch})

	#Show result
	if it%100 == 0:
		print("Iter: {}".format(it))
		print("  G1_loss: {:.4f}, D1_loss: {:.4f},".format(loss_g1, loss_d1))	
		print("  G2_loss: {:.4f}, D2_loss: {:.4f},".format(loss_g2, loss_d2))
		print("  Gs_loss: {:.4f}, Ss_loss: {:.4f}\n".format(loss_gs, loss_ss))
		
		x1_samp, x2_samp = samp_fig(sess, (6,6))
		plot_x(i, x1_samp, x2_samp, (6,6))
		i += 1