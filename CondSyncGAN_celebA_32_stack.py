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

#============================== Parameter ==============================

sketch_dir = 'img_align_celeba_crop_sketch_32/'
celebA_dir = 'img_align_celeba_crop_32/'
output_dir = 'out'
batch_size = 32

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        x1_sync, x2_sync, s_sync = sync_next_batch(x1_train, x2_train, size[1])
        x1_samp[i] = x1_sync
        x2_samp[i] = sess.run(g_sample, feed_dict={x1_: x1_sync, training_: False})

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

#============================== Variable ==============================

def xavier_init(size):
    if len(size) == 4:
        n_inputs = size[0]*size[1]*size[2]
        n_outputs = size[3]
    else:
        n_inputs = size[0]
        n_outputs = size[1]
    
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal(size, stddev=stddev)

def conv2d(x, W, stride=[1,2,2,1], bn=True, is_training=True):
    if bn:
        x = tf.layers.batch_normalization(x, training=is_training)
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride = [1,2,2,1], bn=True, is_training=True):
    if bn:
        x = tf.layers.batch_normalization(x, training=is_training)
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

#============================== Generator ==============================
# 64 * 64
W_G_conv1 = tf.Variable(xavier_init([3,3,1,32]))
b_G_conv1 = tf.Variable(tf.zeros(shape=[32]))
# 64 * 64
W_G_conv2 = tf.Variable(xavier_init([5,5,32,64]))
b_G_conv2 = tf.Variable(tf.zeros(shape=[64]))
# 32 * 32
W_G_conv3 = tf.Variable(xavier_init([5,5,64,128]))
b_G_conv3 = tf.Variable(tf.zeros(shape=[128]))
# 16 * 16
W_G_conv4 = tf.Variable(xavier_init([5,5,128,256]))
b_G_conv4 = tf.Variable(tf.zeros(shape=[256]))
# 8 * 8
W_G_conv5 = tf.Variable(xavier_init([5,5,256,512]))
b_G_conv5 = tf.Variable(tf.zeros(shape=[512]))
# 4 * 4

# W_G_fc1 = tf.Variable(xavier_init([4*4*1024, 4*4*1024]))
# b_G_fc1 = tf.Variable(tf.zeros(shape=[4*4*1024]))

W_G_deconv1 = tf.Variable(xavier_init([5,5,256,512]))
b_G_deconv1 = tf.Variable(tf.zeros(shape=[256]))
# 8 * 8
W_G_deconv2 = tf.Variable(xavier_init([5,5,128,256]))
b_G_deconv2 = tf.Variable(tf.zeros(shape=[128]))
# 16 * 16
W_G_deconv3 = tf.Variable(xavier_init([5,5,64,128]))
b_G_deconv3 = tf.Variable(tf.zeros(shape=[64]))
# 32 * 32
W_G_deconv4 = tf.Variable(xavier_init([5,5,32,64]))
b_G_deconv4 = tf.Variable(tf.zeros(shape=[32]))
# 64 * 64
W_G_deconv5 = tf.Variable(xavier_init([3,3,3,32]))
b_G_deconv5 = tf.Variable(tf.zeros(shape=[3]))
# 64 * 64

var_G = [W_G_conv1, b_G_conv1,
            W_G_conv2, b_G_conv2,
            W_G_conv3, b_G_conv3,
            W_G_conv4, b_G_conv4,
            W_G_conv5, b_G_conv5,
            
            # W_G_fc1, b_G_fc1,

            W_G_deconv1, b_G_deconv1,
            W_G_deconv2, b_G_deconv2,
            W_G_deconv3, b_G_deconv3,
            W_G_deconv4, b_G_deconv4,
            W_G_deconv5, b_G_deconv5]

def generator(X, training):
    h_conv1 = tf.nn.relu(conv2d(X, W_G_conv1, stride=[1,1,1,1], bn=False, is_training=training) + b_G_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_G_conv2, is_training=training) + b_G_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_G_conv3, is_training=training) + b_G_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_G_conv4, is_training=training) + b_G_conv4)
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_G_conv5, is_training=training) + b_G_conv5)

    # h_conv5_re = tf.reshape(h_conv5, [-1, 4*4*1024])
    # h_conv5_re = tf.layers.batch_normalization(h_conv5_re, training=True)
    # h_fc1 = tf.nn.relu(tf.matmul(h_conv5_re, W_G_fc1) + b_G_fc1)
    # h_fc1_re = tf.reshape(h_fc1, [-1, 4, 4, 1024])

    output_shape = tf.stack([tf.shape(X)[0], 4, 4, 256])
    h_deconv1 = tf.nn.relu(deconv2d(h_conv5, W_G_deconv1, output_shape, is_training=training) + b_G_deconv1)
    output_shape = tf.stack([tf.shape(X)[0], 8, 8, 128])
    h_deconv2 = tf.nn.relu(deconv2d(h_deconv1, W_G_deconv2, output_shape, is_training=training) + b_G_deconv2)
    output_shape = tf.stack([tf.shape(X)[0], 16, 16, 64])
    h_deconv3 = tf.nn.relu(deconv2d(h_deconv2, W_G_deconv3, output_shape, is_training=training) + b_G_deconv3)
    output_shape = tf.stack([tf.shape(X)[0], 32, 32, 32])
    h_deconv4 = deconv2d(h_deconv3, W_G_deconv4, output_shape, is_training=training) + b_G_deconv4
    output_shape = tf.stack([tf.shape(X)[0], 32, 32, 3])
    h_deconv5 = deconv2d(h_deconv4, W_G_deconv5, output_shape, stride = [1,1,1,1], is_training=training) + b_G_deconv5

    return tf.nn.sigmoid(h_deconv5)

#============================== Discriminator ==============================
# 64 * 64
W_D_conv1 = tf.Variable(xavier_init([5,5,3,64]))
b_D_conv1 = tf.Variable(tf.zeros(shape=[64]))
# 32 * 32
W_D_conv2 = tf.Variable(xavier_init([5,5,64,128]))
b_D_conv2 = tf.Variable(tf.zeros(shape=[128]))
# 16 * 16
W_D_conv3 = tf.Variable(xavier_init([5,5,128,256]))
b_D_conv3 = tf.Variable(tf.zeros(shape=[256]))
# 8 * 8
W_D_conv4 = tf.Variable(xavier_init([5,5,256,512]))
b_D_conv4 = tf.Variable(tf.zeros(shape=[512]))
# 4 * 4
W_D_fc1 = tf.Variable(xavier_init([2*2*512,512]))
b_D_fc1 = tf.Variable(tf.zeros(shape=[512]))

W_D_fc2 = tf.Variable(xavier_init([512,1]))
b_D_fc2 = tf.Variable(tf.zeros(shape=[1]))

var_D = [W_D_conv1, b_D_conv1,
            W_D_conv2, b_D_conv2,
            W_D_conv3, b_D_conv3,
            W_D_conv4, b_D_conv4,

            W_D_fc1, b_D_fc1,
            W_D_fc2, b_D_fc2]

def discriminator(X, training):
    h_conv1 = tf.nn.relu(conv2d(X, W_D_conv1, bn=False, is_training=training) + b_D_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_D_conv2, is_training=training) + b_D_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_D_conv3, is_training=training) + b_D_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_D_conv4, is_training=training) + b_D_conv4)

    h_conv4_re = tf.reshape(h_conv4, [-1, 2*2*512])
    h_conv4_re = tf.layers.batch_normalization(h_conv4_re, training=True)
    h_fc_1 = tf.nn.relu(tf.matmul(h_conv4_re, W_D_fc1) + b_D_fc1)
    h_fc_1 = tf.layers.batch_normalization(h_fc_1, training=True)
    h_fc_2 = tf.matmul(h_fc_1, W_D_fc2) + b_D_fc2

    return tf.nn.sigmoid(h_fc_2), h_fc_2 

#============================== Synchronizer ==============================
Sync_filter_size = 64
# 64 * 64
W_S_conv1 = tf.Variable(xavier_init([5,5,4,Sync_filter_size]))
b_S_conv1 = tf.Variable(tf.zeros(shape=[Sync_filter_size]))
# 32 * 32
W_S_conv2 = tf.Variable(xavier_init([5,5,Sync_filter_size,Sync_filter_size*2]))
b_S_conv2 = tf.Variable(tf.zeros(shape=[Sync_filter_size*2]))
# 16 * 16
W_S_conv3 = tf.Variable(xavier_init([5,5,Sync_filter_size*2,Sync_filter_size*4]))
b_S_conv3 = tf.Variable(tf.zeros(shape=[Sync_filter_size*4]))
# 8 * 8
W_S_conv4 = tf.Variable(xavier_init([5,5,Sync_filter_size*4,Sync_filter_size*8]))
b_S_conv4 = tf.Variable(tf.zeros(shape=[Sync_filter_size*8]))
# 4 * 4
W_S_fc1 = tf.Variable(xavier_init([2*2*Sync_filter_size*8,512]))
b_S_fc1 = tf.Variable(tf.zeros(shape=[512]))

W_S_fc2 = tf.Variable(xavier_init([512, 1]))
b_S_fc2 = tf.Variable(tf.zeros(shape=[1]))

var_S = [ W_S_conv1, b_S_conv1,
          W_S_conv2, b_S_conv2,
          W_S_conv3, b_S_conv3,
          W_S_conv4, b_S_conv4,
          W_S_fc1, b_S_fc1,
          W_S_fc2, b_S_fc2 ]

def synchronizer(x1, x2, training):
    x12 = tf.concat([x1, x2], axis=3)
    # Mode 1
    h_conv1 = tf.nn.relu(conv2d(x12, W_S_conv1, bn=False, is_training=training) + b_S_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_S_conv2, is_training=training) + b_S_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_S_conv3, is_training=training) + b_S_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_S_conv4, is_training=training) + b_S_conv4)

    h_conv4_re = tf.reshape(h_conv4, [-1, 2*2*Sync_filter_size*8])
    h_conv4_re = tf.layers.batch_normalization(h_conv4_re, training=True)

    # Shared
    h_s1 = tf.nn.relu(tf.matmul(h_conv4_re, W_S_fc1) + b_S_fc1)
    h_s1 = tf.layers.batch_normalization(h_s1, training=True)
    h_s2 = tf.matmul(h_s1, W_S_fc2) + b_S_fc2

    return tf.nn.sigmoid(h_s2), h_s2

#============================== Placeholder & Node ==============================

x1_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
x2_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
s_ = tf.placeholder(tf.float32, shape=[None, 1])
training_ = tf.placeholder(tf.bool)

g_sample = generator(x1_, training_)

D_real_prob, D_real_logit = discriminator(x2_, training_)
D_fake_prob, D_fake_logit = discriminator(g_sample, training_)

S_real_prob, S_real_logit = synchronizer(x1_, x2_, training_)
S_fake_prob, S_fake_logit = synchronizer(x1_, g_sample, training_)

#============================== Loss & Solver ==============================
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

#Solver 
G_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(G_loss, var_list=var_G)
D_solver = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(D_loss, var_list=var_D)

Gs_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(Gs_loss, var_list=var_G)
Ss_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(Ss_loss, var_list=var_S)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#============================== Main ==============================

x1_train, x2_train = read_data(5000)

i=0
for it in range(20001):
	#Get batch training data
	x1_sync, x2_sync, s_sync = sync_next_batch(x1_train, x2_train, batch_size)
	x1_nsync, x2_nsync, s_nsync = nsync_next_batch(x1_train, x2_train, batch_size)
	
	x1_batch = np.concatenate((x1_sync, x1_nsync), axis=0)
	x2_batch = np.concatenate((x2_sync, x2_nsync), axis=0)
	s_batch = np.concatenate((s_sync, s_nsync), axis=0)
	
	#Training
	#_, loss_d = sess.run([D_solver, D_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, training_: True})
	_, loss_ss = sess.run([Ss_solver, Ss_loss], feed_dict={x1_:x1_batch, x2_:x2_batch, s_:s_batch, training_: True})
	
	#_, loss_g = sess.run([G_solver, G_loss], feed_dict={x1_:x1_batch, training_: True})
	#_, loss_gs = sess.run([Gs_solver, Gs_loss], feed_dict={x1_:x1_batch, training_: True})
		
	#Show result
	if it%100 == 0:
		print("Iter: {}".format(it))
		#print("  G_loss : {:.4f}, D_loss : {:.4f}".format(loss_g, loss_d))	
		print("  Gs_loss: {:.4f}, Ss_loss: {:.4f}".format(loss_ss, loss_ss))
		print()
		
		x1_samp, x2_samp = samp_fig(sess, (6,6))
		plot_x(i, x1_samp, x2_samp, (6,6))
		i+=1