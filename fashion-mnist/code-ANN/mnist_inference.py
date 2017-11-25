# coding:utf-8
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784     # 输入层的节点数，对于MNIST数据集，等于图片的像素
OUTPUT_NODE = 10     # 输出层的节点数，这个等于类别的数目，因为MNIST数据集区分0~9，所以输出层节点数为10.
LAYER1_NODE = 784    # 隐藏层一节点数，有784个节点
LAYER2_NODE = 200    # 隐藏层二节点数，有100个节点
BATCH_SIZE = 200     # 每次batch打包的样本个数,数字越小训练过程越接近随机梯度下降，越大越接近梯度下降
REGULARAZTION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数


# 生成各层权值和偏置值
def get_weights_bases(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weights, bases


# 隐藏层运算
def hidden_inference(input_tensor, weights, biases):
    layer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    return layer


# 输出层运算
def output_inference(input_tensor, weights, biases):
    layer = tf.matmul(input_tensor, weights) + biases
    return layer


def inference(x, y_, variable_averages=None):
    # 生成隐藏层1的参数。
    weights1, biases1 = get_weights_bases([INPUT_NODE, LAYER1_NODE])
    # 生成隐藏层2的参数。
    weights2, biases2 = get_weights_bases([LAYER1_NODE, LAYER2_NODE])
    # 生成输出层的参数。
    weights3, biases3 = get_weights_bases([LAYER2_NODE, OUTPUT_NODE])
    # 隐藏层1的输出
    y1 = hidden_inference(x, weights1, biases1)
    # 隐藏层2的输出
    y2 = hidden_inference(y1, weights2, biases2)
    # 输出层结果
    y = output_inference(y2, weights3, biases3)
    # 使用TensorFlow中提供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不是用偏置项
    regularaztion = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
    # 总损失等于交叉熵损失和正则化损失的总和
    loss = cross_entropy_mean + regularaztion
    return y, loss
