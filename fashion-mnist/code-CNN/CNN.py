# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

batch_size = 1024

MODEL_SAVE_PATH = "./model/pb/"


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def get_weights_bases(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weights, bases


def output_inference(input_tensor, weights, biases):
    layer = tf.matmul(input_tensor, weights) + biases
    return layer


def model(X, w, w2, w3, b3, w4, b4, w5, b5, p_keep_conv):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    # shape = [?, 10, 10, 32]

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    l2 = tf.reshape(l2, [-1, w3.get_shape().as_list()[0]])
    l2 = tf.nn.dropout(l2, p_keep_conv)
    # shape = [?, 4, 4, 64]

    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

    layer = tf.matmul(l4, w5) + b5

    return layer


def main():
    with tf.Graph().as_default() as graph:
        mnist = input_data.read_data_sets("../data/fashion", one_hot=True)
        trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
        trX = trX.reshape(-1, 28, 28, 1)
        teX = teX.reshape(-1, 28, 28, 1)
        X = tf.placeholder("float", [None, 28, 28, 1], name='x-input')
        Y = tf.placeholder("float", [None, 10], name='y-input')
        w = init_weight([3, 3, 1, 32])
        w2 = init_weight([3, 3, 32, 64])
        w3, b3 = get_weights_bases([64 * 4 * 4, 300])
        w4, b4 = get_weights_bases([300, 100])
        w5, b5 = get_weights_bases([100, 10])
        p_keep_conv = tf.placeholder("float", name='p_keep_conv')
        py_x = model(X, w, w2, w3, b3, w4, b4, w5, b5, p_keep_conv)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.AdamOptimizer().minimize(cost)
        predict_op = tf.argmax(py_x, 1)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        max_acc = 0.0
        for i in range(100):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                         p_keep_conv: 0.8})
            accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(
                 predict_op, feed_dict={X: teX, p_keep_conv: 1.0}))
            print(i, accuracy)
            if (accuracy > max_acc):
                max_acc = accuracy
                new_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['ArgMax'])
                tf.train.write_graph(new_graph, MODEL_SAVE_PATH, 'graph.pb', as_text=False)


if __name__ == "__main__":
    main()
