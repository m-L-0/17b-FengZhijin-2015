import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784     # 输入层的节点数
OUTPUT_NODE = 10    # 输出层的节点数

mnist = input_data.read_data_sets("../data/fashion", one_hot=True)

with tf.Graph().as_default() as new_graph:
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_')
    with tf.gfile.FastGFile('./model/pb/graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_out = tf.import_graph_def(
            graph_def,
            input_map={'x-input:0': x},
            return_elements=['add_2:0'])
    y = g_out[0]
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session(graph=new_graph) as sess:
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
