import tensorflow as tf
import read_image

mnist = read_image.Dataset(dtype='float32')

with tf.Graph().as_default() as new_graph:
    x = mnist.next_batch(6500)
    x = x.reshape(-1, 28, 28, 1)
    with tf.gfile.FastGFile('./model/pb/graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_out = tf.import_graph_def(
            graph_def,
            input_map={'x-input:0': x,
                       'p_keep_conv:0': 1.0},
            return_elements=['ArgMax:0'])
    y = g_out[0]

with tf.Session(graph=new_graph) as sess:
    y_ = sess.run(y)
    with open('FengZhijin.txt', 'w') as file_object:
        for i in range(6500):
            file_object.write(str(y_[i]) + '\n')

    print(y_)
