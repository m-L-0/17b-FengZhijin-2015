# coding:utf-8
# author: Fengzhijin
# time: 2017.11.20
# ==================================
'''
运用KNN算法解决FashionMNIST数据识别问题
1.loadMNIST() - 数据读取函数
2.KNN() - KNN模型函数
'''


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 模型相关参数
TRAIN_SIZE = 55000    # 训练集数量
TEST_SIZE = 200    # 测试集数量
INPUT_NODE = 784    # 样本特征数
OUTPUT_NODE = 10    # 类别数
MAX = 100.0


# MNIST数据集数据读取
def loadMNIST():
    mnist = input_data.read_data_sets('../data/fashion', one_hot=True)
    return mnist


# KNN模型函数
# 参数：mnist：数据集数据，   k：KNN算法k值
def KNN(mnist, k=1):
    # 训练集数据集和标签集
    train_x = mnist.train.images
    train_y = mnist.train.labels
    # 测试集数据集和标签集
    test_x, test_y = mnist.test.next_batch(TEST_SIZE)

    # 输入训练集和测试集
    xtr = tf.placeholder(tf.float32, [None, INPUT_NODE])
    xte = tf.placeholder(tf.float32, [INPUT_NODE])
    
    # 计算测试集数据与训练集数据的欧式距离
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)), 2), reduction_indices=1))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        right = 0
        # 循环遍历每个测试集数据
        for i in range(TEST_SIZE):
            # 计算测试集数据第i个数据与训练集数据的各点距离
            distances = sess.run(distance, {xtr: train_x, xte: test_x[i, :]})
            # 记录各类结果的概率
            y = np.zeros((1, OUTPUT_NODE))
            for j in range(k):
                pred = np.argmin(distances)
                y += train_y[pred]
                distances[pred] = MAX
            # 打印训练结果
            print('prediction is ', np.argmax(y))
            # 打印正确结果
            print('true value is ', np.argmax(test_y[i]))
            # 判断训练结果是否正确
            if np.argmax(test_y[i]) == np.argmax(y):
                right += 1.0
        # 计算算法正确率
        accracy = right/200.0
        print("k = %d:  %g " % (k, accracy))
    return accracy


def main(argv=None):
    mnist = loadMNIST()
    # 因测试集是对原有测试集的随机选择TEST_SIZE个数据，所以K的值选择不具有普遍性，故此处默认取1
    KNN(mnist)
    #当测试集稳定时此代码可以选出最佳K值
    # couten = []
    # for i in range(50):
    #     couten.append(KNN(mnist, i))
    # print(np.argmax(couten)+1)

if __name__ == "__main__":
    main()
    