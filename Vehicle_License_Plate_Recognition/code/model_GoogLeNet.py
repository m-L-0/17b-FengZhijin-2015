# coding: utf-8
import tensorflow as tf
import read_data as rd
import numpy as np

classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
           8: 'J', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q',
           15: 'R', 16: 'S', 17: 'T', 18: 'U', 19: 'V', 20: 'W', 21: 'X',
           22: 'Y', 23: 'Z', 24: '藏', 25: '川', 26: '鄂', 27: '甘', 28: '赣',
           29: '广', 30: '桂', 31: '贵', 32: '黑', 33: '沪', 34: '吉', 35: '冀',
           36: '津', 37: '晋', 38: '京', 39: '辽', 40: '鲁', 41: '蒙', 42: '闽',
           43: '宁', 44: '青', 45: '琼', 46: '陕', 47: '苏', 48: '皖', 49: '湘',
           50: '新', 51: '渝', 52: '豫', 53: '粤', 54: '云', 55: '浙', 56: '0',
           57: '1', 58: '2', 59: '3', 60: '4', 61: '5', 62: '6', 63: '7',
           64: '8', 65: '9'}
test_size = 630

with tf.Graph().as_default() as new_graph:
    x, y_ = rd.get_test()
    with tf.gfile.FastGFile('./model/pb/GoogLeNet_v3_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_out = tf.import_graph_def(
            graph_def,
            input_map={'x-input:0': x,
                       'p_keep_conv': 1.0},
            return_elements=['ArgMax:0'])
    y = g_out[0]
    y_ = tf.argmax(y_, 1)
    correct_prediction = tf.equal(y, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session(graph=new_graph) as sess:
    predict_y, reality_y = sess.run([y, y_])
    N = np.zeros([66, 66], dtype=int)
    for i in range(630):
        N[reality_y[i]][predict_y[i]] += 1
    for j in range(66):
        rowsum, colsum = sum(N[j]), sum(N[r][j] for r in range(66))
        if (colsum == 0):
            precision = 0
        else:
            precision = N[j][j]/float(colsum)
        if (rowsum == 0):
            recall = 0
        else:
            recall = N[j][j]/float(rowsum)
        print(repr(classes[j]) + "类别的precision=%.2f,recall=%.2f" % (precision, recall))
    print(sess.run(accuracy))