# -*- coding: utf-8 -*-
"""
@CREATETIME: 2018/10/23 13:39 
@AUTHOR: Chans
@VERSION: 1.0
"""
import tensorflow as tf
import tensorboard as tb
import numpy as np
import time
# import sys
#
# sys.path.append(r'/Users/Apple/bbdd/models/tutorials/image/cifar10')
import cifar10, cifar10_input

bath_size = 128
max_steps = 3000
data_dir = '/cifar10_data/cifar-10-batches-bin'

# 需要对权重做一个L2的loss正则约束，特征无效时，会被施加很大的惩罚提升模型泛华能力
# 使用w1对L2的loss控制weights L2的大小
def variable_with_weights_loss(shape, stddv, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddv=stddv))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss, wl, name='weight_loss')
        tf.add_to_collection('losses', weights_loss) # 啥用？？收集在一起
    return var


cifar10.maybe_download_and_extract()
images_train, labels_train = cifar10_input.distorted_inputs(data_dir, bath_size)
images_test, labels_test = cifar10_input.inputs(True, data_dir, bath_size)

image_holder = tf.placeholder(tf.float32, [bath_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [bath_size])


weight1 = variable_with_weights_loss(shape= [5, 5, 3, 64], stddv= 5e-2, wl=.0) # 初始bias设定为多少合适
kernel1 = tf.nn.conv2d(input=image_holder, filter=weight1, strides=[1, 1, 1, 1],
                       padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(input=pool1, depth_radius=4, bias=1.0, alpha=.001/9.0, beta=.75) # 参数设置原因？

weight2 = variable_with_weights_loss(shape=[5, 5, 64, 64], stddv=5e-2, wl=.0)
kernel2 = tf.nn.conv2d(input=norm1, filter=weight2, strides=[1, 1, 1, 1],
                       padding='SAME')
bias2 = tf.Variable(tf.constant(.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(input=conv2, depth_radius=4, bias=1.0, alpha=.001/9.0, beta=.75) # 为啥先进行LRN层处理
pool2 = tf.nn.max_pool(value=norm2, ksize=[1, 2, 2, 1], strides=[1, 3, 3, 1],
                       padding='SAME')

# 需要将数据进行flatten，将样本都变成一维向量
reshape = tf.reshape(tensor=pool2, shape=[bath_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weights_loss(shape=[dim, 384], stddv=.04, wl=.004)
bias3 = tf.Variable(tf.constant(.1, shape=[384])) # 384 = 128 bath* 3 dim
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3) # 因为是全连接层

weight4 = variable_with_weights_loss(shape=[3384, 192], stddv=.04, wl=.004)
bias4 = tf.Variable(tf.constant(.1, shape=[192]))
local4 = tf.nn.relu(features=tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weights_loss(shape=[192, 10], stddv=1/192.0, wl=.0)
bias5 = tf.Variable(tf.constant(.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection(key='losses'), name='total_loss')

loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run(fetches=[images_train, labels_train])
    _, loss_value = sess.run(fetches=[train_op, loss],
                            feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 20 == 0 :
        examples_per_sec = bath_size / duration
        sec_per_batch = float(duration)

        format_str = ('Step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' )
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))


# 测试集内容