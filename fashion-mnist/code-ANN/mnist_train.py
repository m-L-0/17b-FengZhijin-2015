# coding:utf-8
import tensorflow as tf
import mnist_inference as inf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util


INPUT_NODE = 784     # 输入层的节点数
OUTPUT_NODE = 10     # 输出层的节点数
BATCH_SIZE = 100     # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.1    # 初始化的学习率
LEARNING_RATE_DECAY = 0.99    # 学习率的衰减率

TRAINING_STEPS = 100000    # 训练轮数

# 模型持久化
MODEL_SAVE_PATH = "./model/pb/"


def train(mnist):
    with tf.Graph().as_default() as graph:
        # 生成输入数据集和标签集
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        # 定义存储训练轮数的变量
        global_step = tf.Variable(0)
        # 计算前向传播结果和损失函数
        y, loss = inf.inference(x, y_)

        # 设置指数衰减的学习率。
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,    # 初始化的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
            global_step,    # 当前迭代的轮数
            mnist.train.num_examples / BATCH_SIZE,    # 过完所有的训练数据需要的迭代次数
            LEARNING_RATE_DECAY)    # 学习率衰减速度

        # 使用梯度下降优化算法来优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)
        # train_step = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # 更新神经网络中的参数
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')

        # 检验神经网络前向传播结果是否正确
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 计算这一组数据上的正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 将accuracy正确率在tensorboard的SCALAR、HISTOGRAM面板上创建可视化摘要
        summary_op = tf.summary.scalar('value', accuracy)
        histogram = tf.summary.histogram('histogram', accuracy)

        # 模型的初始化
        # saver = tf.train.Saver()

    # 初始化回话并开始训练过程。
    with tf.Session(graph=graph) as sess:
        # 创建事件文件
        writer = tf.summary.FileWriter('./graphs')
        writer.add_graph(graph)

        # 初始化所有变量
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images, y_:
                         mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 记录最大正确率
        max_acc = 0.0

        # 迭代地训练神经网络。
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算模型验证数据上的正确率
                # validate_acc = sess.run(accuracy, feed_dict=validate_feed)

                # 计算模型在验证数据集上的正确率并运行摘要
                summary, acc = sess.run([summary_op, accuracy], feed_dict=validate_feed)
                writer.add_summary(summary, i)
                summaries = sess.run(histogram, feed_dict=validate_feed)
                writer.add_summary(summaries, i)

                print("After %d training step(s), validation accuracy using average model is %g " % (i, acc))

                # 保存训练模型
                if (acc > max_acc):
                    max_acc = acc
                    new_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['add_2'])
                    tf.train.write_graph(new_graph, MODEL_SAVE_PATH, 'graph.pb', as_text=False)

            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

        # 关闭事件文件
        writer.close()


# 主程序
def main(argv=None):
    # 使用tensorflow方法处理数据集
    mnist = input_data.read_data_sets("../data/fashion", one_hot=True)
    train(mnist)


# TensorFlow提供的一个主程序入口
if __name__ == '__main__':
    main()
