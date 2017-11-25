# coding:utf-8
# author: Fengzhijin
# time: 2017.11.22
# ==================================
'''
运用K-means解决FashionMNIST数据识别问题
1.loadMNIST() - 数据读取函数
2.Distance() - 欧氏距离计算函数
3.randCent() - 质心初始化函数
4.Kmeans() - Kmeans算法实现函数
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# MNIST数据集数据读取
def loadMNIST():
    mnist = input_data.read_data_sets('../data/fashion', one_hot=True)
    return mnist


# 计算data1与data2的欧氏距离
def Distance(data1, data2):
    # dist = (data1 - data2).square().sum().sqrt()
    dist = np.sqrt(np.sum(np.square(data1 - data2)))
    return dist


# 初始化k个质心，dataSet为数据集
def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.zeros((k, n))
    for i in range(n):
        minJ = min(dataSet[:, i])
        maxJ = max(dataSet[:, i])
        rangeJ = float(maxJ - minJ)
        centroids[:, i] = minJ + rangeJ * np.random.rand(k)
    return centroids


# Kmeans计算函数
# 输入： mnist：待分类数据集， K：待分类类别数
def Kmeans(mnist, k):
    train_x = mnist.train.images
    # 分类类别及距离记录
    clusterAssment = tf.zeros((train_x.shape[0], 2))
    # 初始化质心
    centroid = tf.constant(randCent(train_x, k), dtype=tf.float32)
    with tf.Session() as sess:
        centroids = sess.run(centroid)      
        clusterassment = sess.run(clusterAssment)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            # 对数据集中每组数据进行分类
            for i in range(train_x.shape[0]):
                minDist = 1000.0
                minIndex = -1
                for j in range(k):
                    distJI = Distance(centroids[j, :], train_x[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterassment[i, 0] != minIndex:
                    clusterChanged = True
                clusterassment[i, :] = minIndex, minDist**2
            # 对质心进行更新
            for cent in range(k):
                ptsInClust = train_x[np.nonzero(clusterassment[:, 0]-cent)[0]]
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
            # 打印质心
            print(centroids)
    return 0


def main(argv=None):
    mnist = loadMNIST()
    Kmeans(mnist, 10)


if __name__ == '__main__':
    main()
