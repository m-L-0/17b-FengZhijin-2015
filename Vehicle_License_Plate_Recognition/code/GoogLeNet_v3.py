# coding:utf-8
# author: Fengzhijin
# time: 2017.12.5
# ==================================
'''
运用GoogLeNet_v3思想解决车牌字符识别问题
1.init_weight() - 生成卷积神经网络卷积核函数
2.get_weights_bases() - 生成神经网络各层权值与偏置值函数
3.inception_0_module_0() - 构建第一个inception module 模块组第一个inception module函数
4.inception_0_module_1() - 构建第一个inception module 模块组第二个inception module函数
5.inception_0_module_2() - 构建第一个inception module 模块组第三个inception module函数
6.inception_0() - 组合第一个inception module 模块组
7.inception_1_module_0() - 构建第二个inception module 模块组第一个inception module函数
8.inception_1_module_1() - 构建第二个inception module 模块组第二个inception module函数
9.inception_1_module_2() - 构建第二个inception module 模块组第三个inception module函数
10.inception_1() - 组合第二个inception module 模块组
11.model（）- 模型网络结构
12.此算法对过拟合进行解决，模型只保存正确率高的情况
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
import read_data as rd

classes_zimu = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                8: 'J', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q',
                15: 'R', 16: 'S', 17: 'T', 18: 'U', 19: 'V', 20: 'W', 21: 'X',
                22: 'Y', 23: 'Z'}
classes_hanzi = {24: '藏', 25: '川', 26: '鄂', 27: '甘', 28: '赣', 29: '广', 30: '桂',
                 31: '贵', 32: '黑', 33: '沪', 34: '吉', 35: '冀', 36: '津', 37: '晋',
                 38: '京', 39: '辽', 40: '鲁', 41: '蒙', 42: '闽', 43: '宁', 44: '青',
                 45: '琼', 46: '陕', 47: '苏', 48: '皖', 49: '湘', 50: '新', 51: '渝',
                 52: '豫', 53: '粤', 54: '云', 55: '浙'}
classes_shuzi = {56: '0', 57: '1', 58: '2', 59: '3', 60: '4', 61: '5', 62: '6',
                 63: '7', 64: '8', 65: '9'}
classes = ['字母', '汉字', '数字']
size = 18449
validation_size = 315
test_size = 630
train_size = 17504

batch_size = 250

MODEL_SAVE_PATH = "./model/pb/"


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def get_weights_bases(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weights, bases


def inception_0_module_0(net):
    branch_0 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_1 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_2 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='SAME')
    branch_3 = tf.nn.relu(tf.nn.conv2d(branch_3, init_weight([3, 3, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    return tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    # shape = [?, 24, 12, 128]


def inception_0_module_1(net):
    branch_0 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_1 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([3, 3, 32, 64]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_2 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='SAME')
    branch_3 = tf.nn.relu(tf.nn.conv2d(branch_3, init_weight([3, 3, 128, 64]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    return tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    # shape = [?, 24, 12, 192]


def inception_0_module_2(net):
    branch_0 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_1 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_2 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 128, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 32, 32]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    branch_3 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='SAME')
    branch_3 = tf.nn.relu(tf.nn.conv2d(branch_3, init_weight([3, 3, 128, 64]),
                          strides=[1, 1, 1, 1], padding='SAME'))

    return tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    # shape = [?, 24, 12, 160]


def inception_0(net):
    return tf.concat([inception_0_module_0(net), inception_0_module_1(net),
                     inception_0_module_2(net)], 3)
    # shape = [?, 24, 12, 480]


def inception_1_module_0(net):
    branch_0 = tf.nn.relu(tf.nn.conv2d(net, init_weight([3, 3, 480, 64]),
                          strides=[1, 2, 2, 1], padding='VALID'))
    # shape = [?, 11, 5, 64]

    branch_1 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 128]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([3, 3, 128, 64]),
                          strides=[1, 2, 2, 1], padding='VALID'))
    # shape = [?, 11, 5, 64]

    branch_2 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 256]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 256, 128]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.relu(tf.nn.conv2d(branch_2, init_weight([3, 3, 128, 64]),
                          strides=[1, 2, 2, 1], padding='VALID'))
    # shape = [?, 11, 5, 64]

    branch_3 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                              padding='VALID')
    branch_3 = tf.nn.relu(tf.nn.conv2d(branch_3, init_weight([1, 1, 480, 128]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    # shape = [?, 11, 5, 128]

    return tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    # shape = [?, 11, 5, 320]


def inception_1_module_1(net):
    branch_0 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 64]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    # shape = [?, 24, 12, 64]

    branch_1 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 256]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([1, 5, 256, 128]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([5, 1, 128, 64]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    # shape = [?, 24, 12, 64]

    branch_2 = tf.concat([
        tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 3, 480, 64]),
                   strides=[1, 1, 1, 1], padding='SAME')),
        tf.nn.relu(tf.nn.conv2d(net, init_weight([3, 1, 480, 64]),
                   strides=[1, 1, 1, 1], padding='SAME'))],
        3)
    # shape = [?, 24, 12, 128]

    return(tf.nn.avg_pool(tf.concat([branch_0, branch_1, branch_2], 3),
                          ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding="VALID"))
    # shape = [?, 11, 5, 256]


def inception_1_module_2(net):

    branch_0 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 256]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_0 = tf.nn.relu(tf.nn.conv2d(branch_0, init_weight([3, 3, 256, 64]),
                          strides=[1, 2, 2, 1], padding='VALID'))
    branch_0 = tf.nn.avg_pool(branch_0, ksize=[1, 3, 3, 1],
                              strides=[1, 1, 1, 1], padding="SAME")
    # shape = [?, 11, 5, 64]

    branch_1 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 256]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([1, 3, 256, 128]),
                          strides=[1, 1, 2, 1], padding='VALID'))
    branch_1 = tf.nn.relu(tf.nn.conv2d(branch_1, init_weight([3, 1, 128, 64]),
                          strides=[1, 2, 1, 1], padding='VALID'))
    # shape = [?, 11, 5, 64]

    branch_2 = tf.nn.relu(tf.nn.conv2d(net, init_weight([1, 1, 480, 128]),
                          strides=[1, 1, 1, 1], padding='SAME'))
    branch_2 = tf.nn.max_pool(branch_2, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    # shape = [?, 11, 5, 128]

    return tf.concat([branch_0, branch_1, branch_2], 3)
    # shape = [?, 11, 5, 256]


def inception_1(net):
    return tf.concat([inception_1_module_0(net),
                      inception_1_module_1(net),
                      inception_1_module_2(net)], 3)
    # shape = [?, 11, 5 , 832]


def model(X, w, b, p_keep_conv):
    l1 = tf.nn.relu(tf.nn.conv2d(X, init_weight([3, 3, 3, 32]),
                    strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.dropout(l1, p_keep_conv)
    # shape = [?, 48, 24, 32]

    l2 = tf.nn.relu(tf.nn.conv2d(l1, init_weight([3, 3, 32, 64]),
                    strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.dropout(l2, p_keep_conv)
    # shape = [?, 48, 24, 64]

    l3 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_conv)
    # shape = [?, 24, 12, 64]

    l4 = tf.nn.relu(tf.nn.conv2d(l3, init_weight([1, 1, 64, 64]),
                    strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.dropout(l4, p_keep_conv)
    # shape = [?, 24, 12, 64]

    l5 = tf.nn.relu(tf.nn.conv2d(l4, init_weight([3, 3, 64, 128]),
                    strides=[1, 1, 1, 1], padding='SAME'))
    l5 = tf.nn.dropout(l5, p_keep_conv)
    # shape = [?, 24, 12, 128]
    # l5 = tf.reshape(l5, [-1, w3.get_shape().as_list()[0]])

    l6 = inception_0(l5)
    l6 = tf.nn.dropout(l6, p_keep_conv)

    l7 = inception_1(l6)
    l7 = tf.nn.dropout(l7, p_keep_conv)
    l7 = tf.reshape(l7, [-1, w.get_shape().as_list()[0]])

    layer = tf.matmul(l7, w) + b

    return layer


def main():
    with tf.Graph().as_default() as graph:
        trX, trY = rd.get_train()
        teX, teY = rd.get_validation()
        X = tf.placeholder("float", [None, 48, 24, 3], name='x-input')
        Y = tf.placeholder("float", [None, 66], name='y-input')
        w, b = get_weights_bases([11*5*832, 66])
        p_keep_conv = tf.placeholder("float", name='p_keep_conv')
        py_x = model(X, w, b, p_keep_conv)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.AdamOptimizer().minimize(cost)
        predict_op = tf.argmax(py_x, 1)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        max_acc = 0.
        for i in range(30):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                tr_op, loss = sess.run([train_op, cost], feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8})
            accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(
                 predict_op, feed_dict={X: teX,  p_keep_conv: 1.0}))
            print(i, accuracy, loss)
            if (accuracy > max_acc):
                max_acc = accuracy
                new_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['ArgMax'])
                tf.train.write_graph(new_graph, MODEL_SAVE_PATH, 'GoogLeNet_v3_graph.pb', as_text=False)


if __name__ == "__main__":
    main()
