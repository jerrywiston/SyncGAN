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
def plot(samples, size, re_size=(32,32,3), gray=False):
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(size[0], size[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample_re = np.reshape(sample, re_size)

        if gray==True:
        	plt.imshow(sample_re, cmap='Greys_r')
        else:
        	plt.imshow(sample_re)
        
    return fig

def plot_mnist(id, type, samp, size=(4,4)):
    fig = plot(samp, size, (28,28), True)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def plot_cifar(id, type, samp, size=(4,4)):
    fig = plot(samp, size, (32,32,3), False)
    plt.savefig('out/{}_{}.png'.format(str(id).zfill(4), type), bbox_inches='tight')
    plt.close(fig)

def samp_fig(sess):
	z1_batch = sample_z(16, z1_dim)
	z2_batch = sample_z(16, z2_dim)
	c_batch = sample_z(16, c_dim)
	x1_samp, x2_samp = sess.run([G1_sample,G2_sample], feed_dict={z1_:z1_batch, z2_: z2_batch, c1_:c_batch, c2_: c_batch})
	return x1_samp, x2_samp

#==================== Data Batch ====================
def class_list(imgs, labels, c=10):
	imgs_list = imgs.tolist()
	imgs_class_list = []
	for i in range(c):
		imgs_class_list.append([])
	
	for i in range(labels.shape[0]):
		imgs_class_list[labels[i]].append(imgs_list[i])

	return imgs_class_list

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
		r = random.randint(0, 500)

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
z1_dim = 8
z2_dim = 16
c_dim = 64

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
z1_ = tf.placeholder(tf.float32, shape=[None, z1_dim])
z2_ = tf.placeholder(tf.float32, shape=[None, z2_dim])

c1_ = tf.placeholder(tf.float32, shape=[None, c_dim])
c2_ = tf.placeholder(tf.float32, shape=[None, c_dim])

x1_ = tf.placeholder(tf.float32, shape=[None, 784])
x2_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

s_ = tf.placeholder(tf.float32, shape=[None, 1])

#==================== Generator ====================
#Generator 1
W_m1_g_fc1 = tf.Variable(xavier_init([z1_dim+c_dim,7*7*64]))
b_m1_g_fc1 = tf.Variable(tf.zeros(shape=[7*7*64]))

W_m1_g_conv2 = tf.Variable(xavier_init([3,3,32,64]))
b_m1_g_conv2 = tf.Variable(tf.zeros(shape=[32]))

W_m1_g_conv3 = tf.Variable(xavier_init([5,5,1,32]))
b_m1_g_conv3 = tf.Variable(tf.zeros(shape=[1]))

var_g1 = [W_m1_g_fc1, b_m1_g_fc1, W_m1_g_conv2, b_m1_g_conv2, W_m1_g_conv3, b_m1_g_conv3]

def Generator1(z, c):
    z_c = tf.concat(axis=1, values=[z, c])
    h_g_fc1 = tf.nn.relu(tf.matmul(z_c, W_m1_g_fc1) + b_m1_g_fc1)
    h_g_re1 = tf.reshape(h_g_fc1, [-1, 7, 7, 64])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 14, 14, 32])
    h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_m1_g_conv2, output_shape_g2) + b_m1_g_conv2)

    output_shape_g3 = tf.stack([tf.shape(z)[0], 28, 28, 1])
    h_g_conv3 = tf.nn.sigmoid(deconv2d(h_g_conv2, W_m1_g_conv3, output_shape_g3) + b_m1_g_conv3)

    h_g_re3 = tf.reshape(h_g_conv3, [-1,784])
    return h_g_re3

#Generator 2
W_m2_g_fc1 = tf.Variable(xavier_init([z2_dim+c_dim,4*4*128]))
b_m2_g_fc1 = tf.Variable(tf.zeros(shape=[4*4*128]))

W_m2_g_conv2 = tf.Variable(xavier_init([3,3,64,128]))
b_m2_g_conv2 = tf.Variable(tf.zeros(shape=[64]))

W_m2_g_conv3 = tf.Variable(xavier_init([3,3,32,64]))
b_m2_g_conv3 = tf.Variable(tf.zeros(shape=[32]))

W_m2_g_conv4 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_g_conv4 = tf.Variable(tf.zeros(shape=[16]))

W_m2_g_conv5 = tf.Variable(xavier_init([5,5,3,16]))
b_m2_g_conv5 = tf.Variable(tf.zeros(shape=[3]))

var_g2 = [W_m2_g_fc1, b_m2_g_fc1, 
		 W_m2_g_conv2, b_m2_g_conv2, 
		 W_m2_g_conv3, b_m2_g_conv3, 
		 W_m2_g_conv4, b_m2_g_conv4,
		 W_m2_g_conv5, b_m2_g_conv5]

def Generator2(z, c):
    z_c = tf.concat(axis=1, values=[z, c])
    h_g_fc1 = tf.nn.relu(tf.matmul(z_c, W_m2_g_fc1) + b_m2_g_fc1)
    h_g_re1 = tf.reshape(h_g_fc1, [-1, 4, 4, 128])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 8, 8, 64])
    h_g_conv2 = tf.nn.relu(deconv2d(h_g_re1, W_m2_g_conv2, output_shape_g2) + b_m2_g_conv2)

    output_shape_g3 = tf.stack([tf.shape(z)[0], 16, 16, 32])
    h_g_conv3 = tf.nn.relu(deconv2d(h_g_conv2, W_m2_g_conv3, output_shape_g3) + b_m2_g_conv3)

    output_shape_g4 = tf.stack([tf.shape(z)[0], 16, 16, 16])
    h_g_conv4 = tf.nn.relu(deconv2d(h_g_conv3, W_m2_g_conv4, output_shape_g4, stride=[1,1,1,1]) + b_m2_g_conv4)

    output_shape_g5 = tf.stack([tf.shape(z)[0], 32, 32, 3])
    h_g_conv5 = tf.nn.sigmoid(deconv2d(h_g_conv4, W_m2_g_conv5, output_shape_g5) + b_m2_g_conv5)

    return h_g_conv5

#==================== Discriminator ====================
#Discriminator 1
W_m1_d_conv1 = tf.Variable(xavier_init([5,5,1,4]))
b_m1_d_conv1 = tf.Variable(tf.zeros(shape=[4]))

W_m1_d_conv2 = tf.Variable(xavier_init([3,3,4,8]))
b_m1_d_conv2 = tf.Variable(tf.zeros(shape=[8]))

W_m1_d_fc3 = tf.Variable(xavier_init([7*7*8, 64]))
b_m1_d_fc3 = tf.Variable(tf.zeros(shape=[64]))

W_m1_d_fc4 = tf.Variable(xavier_init([64, 1]))
b_m1_d_fc4 = tf.Variable(tf.zeros(shape=[1]))

var_d1 = [W_m1_d_conv1, b_m1_d_conv1, W_m1_d_conv2, b_m1_d_conv2, W_m1_d_fc3, b_m1_d_fc3, W_m1_d_fc4, b_m1_d_fc4]

def Discriminator1(x):
	x_re = tf.reshape(x, [-1,28,28,1])
	h_d_conv1 = tf.nn.relu(conv2d(x_re, W_m1_d_conv1, [1,2,2,1]) + b_m1_d_conv1)

	h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_m1_d_conv2, [1,2,2,1]) + b_m1_d_conv2)
	h_d_re2 = tf.reshape(h_d_conv2, [-1,7*7*8])

	h_d_fc3 = tf.nn.relu(tf.matmul(h_d_re2, W_m1_d_fc3) + b_m1_d_fc3)
	
	y_logit = tf.matmul(h_d_fc3, W_m1_d_fc4) + b_m1_d_fc4
	y_prob = tf.nn.sigmoid(y_logit)
	
	return y_logit, y_prob

#Discriminator 2
W_m2_d_conv1 = tf.Variable(xavier_init([5,5,3,16]))
b_m2_d_conv1 = tf.Variable(tf.zeros(shape=[16]))

W_m2_d_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_d_conv2 = tf.Variable(tf.zeros(shape=[32]))

W_m2_d_conv3 = tf.Variable(xavier_init([3,3,32,64]))
b_m2_d_conv3 = tf.Variable(tf.zeros(shape=[64]))

W_m2_d_fc4 = tf.Variable(xavier_init([4*4*64, 256]))
b_m2_d_fc4 = tf.Variable(tf.zeros(shape=[256]))

W_m2_d_fc5 = tf.Variable(xavier_init([256, 1]))
b_m2_d_fc5 = tf.Variable(tf.zeros(shape=[1]))

var_d2 = [W_m2_d_conv1, b_m2_d_conv1, 
		 W_m2_d_conv2, b_m2_d_conv2, 
		 W_m2_d_conv3, b_m2_d_conv3, 
		 W_m2_d_fc4, b_m2_d_fc4, 
		 W_m2_d_fc5, b_m2_d_fc5]

def Discriminator2(x):
    h_d_conv1 = tf.nn.relu(conv2d(x, W_m2_d_conv1, [1,2,2,1]) + b_m2_d_conv1)
    h_d_conv2 = tf.nn.relu(conv2d(h_d_conv1, W_m2_d_conv2, [1,2,2,1]) + b_m2_d_conv2)
    h_d_conv3 = tf.nn.relu(conv2d(h_d_conv2, W_m2_d_conv3, [1,2,2,1]) + b_m2_d_conv3)

    h_d_re3 = tf.reshape(h_d_conv3, [-1,4*4*64])
    h_d_fc4 = tf.nn.relu(tf.matmul(h_d_re3, W_m2_d_fc4) + b_m2_d_fc4)

    y_logit = tf.matmul(h_d_fc4, W_m2_d_fc5) + b_m2_d_fc5
    y_prob = tf.nn.sigmoid(y_logit)

    return y_logit, y_prob

#==================== Synchronizer ====================
#Modal 1
W_m1_s1 = tf.Variable(xavier_init([784,256]))
b_m1_s1 = tf.Variable(tf.zeros(shape=[256]))

#Modal 2
W_m2_s_conv1 = tf.Variable(xavier_init([5,5,3,16]))
b_m2_s_conv1 = tf.Variable(tf.zeros(shape=[16]))

W_m2_s_conv2 = tf.Variable(xavier_init([3,3,16,32]))
b_m2_s_conv2 = tf.Variable(tf.zeros(shape=[32]))

W_m2_s_conv3 = tf.Variable(xavier_init([3,3,32,64]))
b_m2_s_conv3 = tf.Variable(tf.zeros(shape=[64]))

W_m2_s_fc4 = tf.Variable(xavier_init([4*4*64, 256]))
b_m2_s_fc4 = tf.Variable(tf.zeros(shape=[256]))

#Shared
W_s_s2 = tf.Variable(xavier_init([512,256]))
b_s_s2 = tf.Variable(tf.zeros(shape=[256]))

W_s_s3 = tf.Variable(xavier_init([256,1]))
b_s_s3 = tf.Variable(tf.zeros(shape=[1]))

var_s = [ W_m1_s1, b_m1_s1, 
		  W_m2_s_conv1, b_m2_s_conv1, 
		  W_m2_s_conv2, b_m2_s_conv2, 
		  W_m2_s_conv3, b_m2_s_conv3, 
		  W_m2_s_fc4, b_m2_s_fc4,
		  W_s_s2, b_s_s2,
		  W_s_s3, b_s_s3 ]

def Synchronizer(x1, x2):
    #Modal 1
    h_m1_s1 = tf.nn.relu(tf.matmul(x1, W_m1_s1) + b_m1_s1)

    #Modal 2
    h_m2_s_conv1 = tf.nn.relu(conv2d(x2, W_m2_s_conv1, [1,2,2,1]) + b_m2_s_conv1)
    h_m2_s_conv2 = tf.nn.relu(conv2d(h_m2_s_conv1, W_m2_s_conv2, [1,2,2,1]) + b_m2_s_conv2)
    h_m2_s_conv3 = tf.nn.relu(conv2d(h_m2_s_conv2, W_m2_s_conv3, [1,2,2,1]) + b_m2_s_conv3)

    h_m2_s_re3 = tf.reshape(h_m2_s_conv3, [-1,4*4*64])
    h_m2_s_fc4 = tf.nn.relu(tf.matmul(h_m2_s_re3, W_m2_s_fc4) + b_m2_s_fc4)

    #Shared
    h_concat_s1 = tf.concat(axis=1, values=[h_m1_s1, h_m2_s_fc4])
    h_s2 = tf.nn.relu(tf.matmul(h_concat_s1, W_s_s2) + b_s_s2)

    y_s_logit = tf.matmul(h_s2, W_s_s3) + b_s_s3
    y_s_prob = tf.nn.sigmoid(y_s_logit)
    return y_s_logit, y_s_prob

G1_sample = Generator1(z1_, c1_)
G2_sample = Generator2(z2_, c2_)

D1_real_logit, D1_real_prob = Discriminator1(x1_)
D1_fake_logit, D1_fake_prob = Discriminator1(G1_sample)

D2_real_logit, D2_real_prob = Discriminator2(x2_)
D2_fake_logit, D2_fake_prob = Discriminator2(G2_sample)

S_real_logit, S_real_prob = Synchronizer(x1_, x2_)
S_fake_logit, S_fake_prob = Synchronizer(G1_sample, G2_sample)

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
Ss_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_real_logit, labels=s_))
Ss_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=tf.zeros_like(S_fake_logit)))
Ss_loss = Ss_real_loss #+ Ss_fake_loss
Gs_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=S_fake_logit, labels=s_))

#Solver 
G1_solver = tf.train.AdamOptimizer().minimize(G1_loss, var_list=var_g1)
G2_solver = tf.train.AdamOptimizer().minimize(G2_loss, var_list=var_g2)

D1_solver = tf.train.AdamOptimizer().minimize(D1_loss, var_list=var_d1)
D2_solver = tf.train.AdamOptimizer().minimize(D2_loss, var_list=var_d2)

Ss_solver = tf.train.AdamOptimizer().minimize(0.2*Ss_loss, var_list=var_s)
Gs_solver = tf.train.AdamOptimizer().minimize(0.2*Gs_loss, var_list=var_g1 + var_g2)
G2s_solver = tf.train.AdamOptimizer().minimize(0.2*Gs_loss, var_list=var_g2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#==================== cifar handling ====================
def labels_one_hot(labels_raw, dim):
    labels = np.zeros((len(labels_raw), dim))
    for i in range(len(labels_raw)):
        labels[i][labels_raw[i]] = 1.
    return labels

def cifar_read(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pinckle = pickle.load(fo, encoding='bytes')
    return dict

def cifar_data_extract(dict):
    imgs_raw = dict[b'data'].astype("uint8")
    imgs = np.array(imgs_raw, dtype=float) / 255.0   
    imgs = imgs.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    labels = np.array(dict[b'labels'], dtype=int)
    #labels = labels_one_hot(dict[b'labels'], 10)
    return imgs, labels

def cifar_next_batch(imgs, labels, size):
    img_samp = np.ndarray(shape=(size, 32, 32 ,3))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        r = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[r]
        label_samp[i] = labels[r]
    return img_samp, label_samp

#==================== Dataset ====================
#Read MNIST
mnist_digit = input_data.read_data_sets('MNIST_digit', one_hot=False)
x_digit = mnist_digit.train.images
y_digit = mnist_digit.train.labels
x1_train = class_list(x_digit, y_digit, 10)

dict1 = cifar_read("cifar-10-batches-py/data_batch_1")
dict2 = cifar_read("cifar-10-batches-py/data_batch_2")
dict3 = cifar_read("cifar-10-batches-py/data_batch_3")
dict4 = cifar_read("cifar-10-batches-py/data_batch_4")
dict5 = cifar_read("cifar-10-batches-py/data_batch_5")

#Read CIFAR-10
x_cifar_p1, y_cifar_p1 = cifar_data_extract(dict1)
x_cifar_p2, y_cifar_p2 = cifar_data_extract(dict2)
x_cifar_p3, y_cifar_p3 = cifar_data_extract(dict3)
x_cifar_p4, y_cifar_p4 = cifar_data_extract(dict4)
x_cifar_p5, y_cifar_p5 = cifar_data_extract(dict5)

x_cifar = np.concatenate((x_cifar_p1, x_cifar_p2), axis=0)
y_cifar = np.concatenate((y_cifar_p1, y_cifar_p2), axis=0)
x2_train = class_list(x_cifar_p1, y_cifar_p1, 10)

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

	z1_batch = sample_z(batch_size*2, z1_dim)
	z2_batch = sample_z(batch_size*2, z2_dim)

	c_sync_batch = sample_z(batch_size, c_dim)
	c1_nsync_batch = sample_z(batch_size, c_dim)
	c2_nsync_batch = sample_z(batch_size, c_dim)

	c1_batch = np.concatenate((c_sync_batch, c1_nsync_batch), axis=0)
	c2_batch = np.concatenate((c_sync_batch, c2_nsync_batch), axis=0)
	sf_batch = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=0)

	#Training
	if it < 15000:
		_, loss_d1 = sess.run([D1_solver, D1_loss], feed_dict={z1_:z1_batch, c1_:c1_batch, x1_:x1_batch})
	else:
		loss_d1 = 0.
	_, loss_d2 = sess.run([D2_solver, D2_loss], feed_dict={z2_:z2_batch, c2_:c2_batch, x2_:x2_batch})
	_, loss_ss = sess.run([Ss_solver, Ss_loss], feed_dict={z1_:z1_batch, z2_:z2_batch, c1_:c1_batch, c2_:c2_batch, x1_:x1_batch, x2_:x2_batch, s_:sr_batch})

	if it < 15000:
		_, loss_g1 = sess.run([G1_solver, G1_loss], feed_dict={z1_:z1_batch, c1_:c1_batch})
		_, loss_gs = sess.run([Gs_solver, Gs_loss], feed_dict={z1_:z1_batch, z2_:z2_batch, c1_:c1_batch, c2_:c2_batch, s_:sf_batch})
	else:
		loss_g1 = 0.
		_, loss_gs = sess.run([G2s_solver, G2s_loss], feed_dict={z1_:z1_batch, z2_:z2_batch, c1_:c1_batch, c2_:c2_batch, s_:sf_batch})
	_, loss_g2 = sess.run([G2_solver, G2_loss], feed_dict={z2_:z2_batch, c2_:c2_batch})
		
	#Show result
	if it%100 == 0:
		print("Iter: {}".format(it))
		print("  G1_loss: {:.4}, G2_loss: {:.4},".format(loss_g1, loss_g2))	
		print("  D1_loss: {:.4}, D2_loss: {:.4},".format(loss_d1, loss_d2))
		print("  Ss_loss: {:.4}, Gs_loss: {:.4}\n".format(loss_ss, loss_gs))
		
		x1_samp, x2_samp = samp_fig(sess)
		plot_mnist(i,'_m1_samp', x1_samp)
		plot_cifar(i,'_m2_samp', x2_samp)
		i += 1