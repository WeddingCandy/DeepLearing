# -*- coding: utf-8 -*-
"""
@CREATETIME: 2018/10/6 17:38 
@AUTHOR: Chans
@VERSION: 
"""
import warnings
warnings.filterwarnings("ignore")
import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/" , one_hot=True)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None, 784]) #图像输入向量
W = tf.Variable(tf.zeros([784,10]))  #权重，初始化值为全零
b = tf.Variable(tf.zeros([10]))  #偏置，初始化值为全零

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) ,reduction_indices= [1]))

learning_rate  = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

tf.global_variables_initializer().run()


for i in range(1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys })


correct_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.train.images,y_ : mnist.test.labels}))
# x = tf.placeholder(tf.float32,[None,in_units] , name = "x_input")

# print(mnist.train.images.shape,mnist.train.labels.shape)
#
