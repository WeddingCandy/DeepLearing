from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())




def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                            dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel,strides= [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1,depth_radius=4, bias=1.0, alpha=.001/9, beta=.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1,3, 3, 1], strides=[1,2,2,1],
                           padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                            dtype=tf.float32, stddev=1e-1),name='weights')
        conv_mid = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='weights')
        biases = tf.Variable(tf.constant(.0, shape=[192],dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv_mid, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2,depth_radius=4, bias=1.0, alpha=.001/9, beta=.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                            dtype=tf.float32, stddev=1e-1),name='weights')
        conv_mid = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='weights')
        biases = tf.Variable(tf.constant(.0, shape=[384],dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv_mid, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)