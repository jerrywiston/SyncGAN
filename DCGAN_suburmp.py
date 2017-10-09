import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import math
from PIL import Image

def plot(samples):
    #samples = (samples + 1.)/2.
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64,64,3))

    return fig

def plot_x(id, type, samp):
    fig = plot(samp)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def next_batch(imgs, labels, size):
    img_samp = np.ndarray(shape=(size, imgs.shape[1]))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        r = random.randint(0,imgs.shape[0]-1)
        img_samp[i] = imgs[r]
        label_samp[i] = labels[r]
    return img_samp, label_samp

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

#================================= Sub-URMP =================================
data_dir = 'Sub_URMP_crop_64/'
ins_list = ['viola' , 'oboe', 'bassoon', 'flute', 'tuba', 'horn', 'sax', 
'double_bass', 'cello', 'trombone', 'violin', 'clarinet', 'trumpet']

def read_data(size, path):
    urmp_list = np.zeros([size, 64, 64, 3], dtype=np.float32)
    
    for i in range(size):
        fname = str(i).zfill(4) + '.png'
        urmp_list[i] = np.array(Image.open(path + fname)) / 255.
        
    return urmp_list

def read_urmp_img(ins_list, path, size):
	x_samp = []
	for ins in ins_list:
		full_path = path + ins + "/img/"
		print(full_path)
		x_temp = read_data(size, full_path)
		x_samp.append(x_temp)

	return np.asarray(x_samp).reshape(-1,64,64,3)

def next_batch(imgs, size):
    img_samp = np.ndarray(shape=(size, 64, 64 ,3))
    for i in range(size):
        r = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[r]
    return img_samp

#===================================================================================
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
        x = batch_normalization(x, is_training)
    return tf.nn.conv2d(x ,W ,strides=stride, padding='SAME')

def deconv2d(x, W, output_shape, stride = [1,2,2,1], bn=True, is_training=True):
    if bn:
        x = batch_normalization(x, is_training)
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=stride, padding='SAME')

def batch_normalization(x, is_training=True):
    return tf.contrib.layers.batch_norm(
            x,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            is_training=is_training
            )

#Parameter
z_dim = 64
batch_size = 128

#Placeholder
z_ = tf.placeholder(tf.float32, shape=[None, z_dim])
x_ = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
training_ = tf.placeholder(tf.bool)

#Generator
W_g_fc1 = tf.Variable(xavier_init([z_dim,4*4*1024]))
b_g_fc1 = tf.Variable(tf.zeros(shape=[4*4*1024]))

W_g_conv2 = tf.Variable(xavier_init([5,5,512,1024]))
b_g_conv2 = tf.Variable(tf.zeros(shape=[512]))

W_g_conv3 = tf.Variable(xavier_init([5,5,256,512]))
b_g_conv3 = tf.Variable(tf.zeros(shape=[256]))

W_g_conv4 = tf.Variable(xavier_init([5,5,128,256]))
b_g_conv4 = tf.Variable(tf.zeros(shape=[128]))

W_g_conv5 = tf.Variable(xavier_init([5,5,64,128]))
b_g_conv5 = tf.Variable(tf.zeros(shape=[64]))

W_g_conv6 = tf.Variable(xavier_init([3,3,3,64]))
b_g_conv6 = tf.Variable(tf.zeros(shape=[3]))

var_g = [W_g_fc1, b_g_fc1, 
         W_g_conv2, b_g_conv2, 
         W_g_conv3, b_g_conv3, 
         W_g_conv4, b_g_conv4, 
         W_g_conv5, b_g_conv5, 
         W_g_conv6, b_g_conv6]

def Generator(z, training):
    h_g_fc1 = tf.nn.relu(tf.matmul(z, W_g_fc1) + b_g_fc1)
    h_g_re1 = tf.reshape(h_g_fc1, [-1, 4, 4, 1024])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 8, 8, 512])
    h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_g_conv2, output_shape_g2, is_training=training) + b_g_conv2)

    output_shape_g3 = tf.stack([tf.shape(z)[0], 16, 16, 256])
    h_g_conv3 = tf.nn.relu(deconv2d(h_g_conv2, W_g_conv3, output_shape_g3, is_training=training) + b_g_conv3)

    output_shape_g4 = tf.stack([tf.shape(z)[0], 32, 32, 128])
    h_g_conv4 = tf.nn.relu(deconv2d(h_g_conv3, W_g_conv4, output_shape_g4, is_training=training) + b_g_conv4)

    output_shape_g5 = tf.stack([tf.shape(z)[0], 64, 64, 64])
    h_g_conv5 = tf.nn.relu(deconv2d(h_g_conv4, W_g_conv5, output_shape_g5, is_training=training) + b_g_conv5)

    output_shape_g6 = tf.stack([tf.shape(z)[0], 64, 64, 3])
    h_g_conv6 = tf.nn.sigmoid(deconv2d(h_g_conv5, W_g_conv6, output_shape_g6, stride=[1,1,1,1], is_training=training) + b_g_conv6)

    return h_g_conv6

#Discriminator
W_d_conv1 = tf.Variable(xavier_init([5,5,3,128]))
b_d_conv1 = tf.Variable(tf.zeros(shape=[128]))

W_d_conv2 = tf.Variable(xavier_init([5,5,128,256]))
b_d_conv2 = tf.Variable(tf.zeros(shape=[256]))

W_d_conv3 = tf.Variable(xavier_init([5,5,256,512]))
b_d_conv3 = tf.Variable(tf.zeros(shape=[512]))

W_d_conv4 = tf.Variable(xavier_init([5,5,512,1024]))
b_d_conv4 = tf.Variable(tf.zeros(shape=[1024]))

W_d_fc5 = tf.Variable(xavier_init([1024,1]))
b_d_fc5 = tf.Variable(tf.zeros(shape=[1]))

var_d = [W_d_conv1, b_d_conv1, W_d_conv2, b_d_conv2, W_d_conv3, b_d_conv3, W_d_conv4, b_d_conv4, W_d_fc5, b_d_fc5]

def Discriminator(x, training):
    h_d_conv1 = tf.nn.relu(conv2d(x, W_d_conv1, [1,2,2,1], bn=False, is_training=training) + b_d_conv1)
    h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_d_conv2, [1,2,2,1], is_training=training) + b_d_conv2)
    h_d_conv3 = tf.nn.relu(conv2d(h_d_conv2, W_d_conv3, [1,2,2,1], is_training=training) + b_d_conv3)
    h_d_conv4 = tf.nn.relu(conv2d(h_d_conv3, W_d_conv4, [1,2,2,1], is_training=training) + b_d_conv4)

    avg_pool = tf.reduce_mean(h_d_conv4, [1, 2])

    y_logit = tf.matmul(avg_pool, W_d_fc5) + b_d_fc5
    y_prob = tf.nn.sigmoid(y_logit)

    return y_prob, y_logit

G_sample = Generator(z_, training_)
D_real, D_logit_real = Discriminator(x_, training_)
D_fake, D_logit_fake = Discriminator(G_sample, training_)

#W-GAN Loss
'''
eps = 1e-8
D_loss = -tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps))
G_loss = -tf.reduce_mean(tf.log(D_fake + eps))
'''
#Vanilla GAN Loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(D_loss, var_list=var_d)
G_solver = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(G_loss, var_list=var_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Main
if not os.path.exists('out/'):
    os.makedirs('out/')

x_train = read_urmp_img(ins_list, data_dir, 1000)
print(x_train.shape)

i=0
for it in range(100001):
    #Train weight & latent
    x_batch = next_batch(x_train, batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x_: x_batch, z_: sample_z(batch_size, z_dim), training_: True})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z_: sample_z(batch_size, z_dim), training_: True})

    #Show result
    if it % 100 == 0:
        print('Iter: {}, G_loss: {:.4}, D_loss: {:.4}'.format(it, G_loss_curr, D_loss_curr))
        z_samp = sample_z(16, z_dim)
        x_samp = sess.run(G_sample, feed_dict={z_: z_samp, training_: False})
        plot_x(i,'samp', x_samp)
        i += 1
