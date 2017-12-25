# coding: utf-8

import tensorflow as tf
import numpy as np
from PIL import Image


def read_data(file):
    image = Image.open(file)
    image = image.resize((50, 40))
    return image


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.11))


def y_reduction(index):
    index_ = list()
    for i in range(len(index)):
        xiabiao = np.argmax(index[i], axis=1)
        out_index = ''
        for j in range(len(xiabiao)):
            if xiabiao[j] != 10:
                out_index = out_index + str(xiabiao[j])
        index_.append(int(out_index))
    return np.array(index_)


def get_weights_bases(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weights, bases


def get_weights_bases_wout(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1], shape[2]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[0], shape[2]]))
    return weights, bases


def model(X, w, w1, w2, w3, w4, w5, b5, w6, b6, p_keep_conv, p_keep_within):
    la = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l = tf.nn.avg_pool(la, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l = tf.nn.dropout(l, p_keep_conv)
    # shape = [?, 20, 30, 32]

    l1a = tf.nn.relu(tf.nn.conv2d(l, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.avg_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    # shape = [?, 10, 15, 64]

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.avg_pool(l2a, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_within)
    # shape = [?, 4, 5, 64]

    l3a_0 = tf.nn.relu(tf.nn.conv2d(l2, w3[0], strides=[1, 1, 1, 1], padding='SAME'))
    l3_0 = tf.nn.avg_pool(l3a_0, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_0 = tf.nn.dropout(l3_0, p_keep_within)
    l3a_1 = tf.nn.relu(tf.nn.conv2d(l2, w3[1], strides=[1, 1, 1, 1], padding='SAME'))
    l3_1 = tf.nn.avg_pool(l3a_1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_1 = tf.nn.dropout(l3_1, p_keep_within)
    l3a_2 = tf.nn.relu(tf.nn.conv2d(l2, w3[2], strides=[1, 1, 1, 1], padding='SAME'))
    l3_2 = tf.nn.avg_pool(l3a_2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_2 = tf.nn.dropout(l3_2, p_keep_within)
    l3a_2 = tf.nn.relu(tf.nn.conv2d(l2, w3[2], strides=[1, 1, 1, 1], padding='SAME'))
    l3_2 = tf.nn.avg_pool(l3a_2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_2 = tf.nn.dropout(l3_2, p_keep_within)
    l3a_3 = tf.nn.relu(tf.nn.conv2d(l2, w3[3], strides=[1, 1, 1, 1], padding='SAME'))
    l3_3 = tf.nn.avg_pool(l3a_3, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_3 = tf.nn.dropout(l3_3, p_keep_within)
    # shape = [?, 4, 5, 64]

    l4a_0 = tf.nn.relu(tf.nn.conv2d(l3_0, w4[0], strides=[1, 1, 1, 1], padding='SAME'))
    l4_0 = tf.reshape(l4a_0, [-1, w5.get_shape().as_list()[1]])
    l4_0 = tf.nn.dropout(l4_0, p_keep_within)
    l4a_1 = tf.nn.relu(tf.nn.conv2d(l3_1, w4[1], strides=[1, 1, 1, 1], padding='SAME'))
    l4_1 = tf.reshape(l4a_1, [-1, w5.get_shape().as_list()[1]])
    l4_1 = tf.nn.dropout(l4_1, p_keep_within)
    l4a_2 = tf.nn.relu(tf.nn.conv2d(l3_2, w4[2], strides=[1, 1, 1, 1], padding='SAME'))
    l4_2 = tf.reshape(l4a_2, [-1, w5.get_shape().as_list()[1]])
    l4_2 = tf.nn.dropout(l4_2, p_keep_within)
    l4a_3 = tf.nn.relu(tf.nn.conv2d(l3_3, w4[3], strides=[1, 1, 1, 1], padding='SAME'))
    l4_3 = tf.reshape(l4a_3, [-1, w5.get_shape().as_list()[1]])
    l4_3 = tf.nn.dropout(l4_3, p_keep_within)
    # shape = [?, 4, 5, 64]

    l5_0 = tf.nn.relu(tf.matmul(l4_0, w5[0]) + b5[0])
    l5_0 = tf.nn.dropout(l5_0, p_keep_conv)
    l5_1 = tf.nn.relu(tf.matmul(l4_1, w5[1]) + b5[1])
    l5_1 = tf.nn.dropout(l5_1, p_keep_conv)
    l5_2 = tf.nn.relu(tf.matmul(l4_2, w5[2]) + b5[2])
    l5_2 = tf.nn.dropout(l5_2, p_keep_conv)
    l5_3 = tf.nn.relu(tf.matmul(l4_3, w5[3]) + b5[3])
    l5_3 = tf.nn.dropout(l5_3, p_keep_conv)

    layer_0 = tf.matmul(l5_0, w6[0]) + b6[0]
    layer_1 = tf.matmul(l5_1, w6[1]) + b6[1]
    layer_2 = tf.matmul(l5_2, w6[2]) + b6[2]
    layer_3 = tf.matmul(l5_3, w6[3]) + b6[3]

    layer = tf.concat([layer_0, layer_1, layer_2, layer_3], 1)
    return tf.reshape(layer, shape=(-1, 4, 11))


def model_test(file):
    with tf.Graph().as_default() as g:
        X = tf.placeholder("float", [None, 40, 50, 3], name='x-input')
        w = init_weight([3, 3, 3, 32])
        w1 = init_weight([3, 3, 32, 64])
        w2 = init_weight([3, 3, 64, 64])
        w3 = init_weight([4, 3, 3, 64, 64])
        w4 = init_weight([4, 3, 3, 64, 64])
        w5, b5 = get_weights_bases_wout([4, 4*5*64, 400])
        w6, b6 = get_weights_bases_wout([4, 400, 11])
        p_keep_conv = tf.placeholder("float", name='p_keep_conv')
        p_keep_within = tf.placeholder("float", name='P_keep_within')
        py_x = model(X, w, w1, w2, w3, w4, w5, b5, w6, b6, p_keep_conv, p_keep_within)

        predict_op = py_x
        saver = tf.train.Saver()

    with tf.Session(graph=g) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './model/model.cpkt-247')
        teX = np.array(read_data(file)).reshape(-1, 40, 50, 3)
        yuce_Y = sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0,
                          p_keep_within: 1.0})
        yc_Y = y_reduction(yuce_Y)
        return yc_Y
