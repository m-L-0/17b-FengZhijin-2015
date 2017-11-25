import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/fashion", one_hot=True)

with tf.Graph().as_default() as new_graph:
    x = mnist.test.images
    x = x.reshape(-1, 28, 28, 1)
    y_ = mnist.test.labels
    with tf.gfile.FastGFile('./model/pb/graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_out = tf.import_graph_def(
            graph_def,
            input_map={'x-input:0': x,
                       'p_keep_conv:0': 1.0},
            return_elements=['ArgMax:0'])
    y = g_out[0]
    correct_prediction = tf.equal(y, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session(graph=new_graph) as sess:
    print(sess.run(accuracy))
