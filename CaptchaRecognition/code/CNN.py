# coding: utf-8

import tensorflow as tf
import numpy as np
import read_data as rd

batch_size = 500


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.11))


def y_handle(index):
    for i in range(len(index)):
        index_ = np.zeros((1, 4, 11), dtype=float)
        for j in range(4):
            xiabiao = 10
            if j < len(str(index[i])):
                xiabiao = int(str(index[i])[j])
            index_[0, j, xiabiao] = 1.0
        if i == 0:
            out_index = np.array(index_)
        else:
            out_index = np.concatenate((out_index, index_), axis=0)
    return out_index


def y_reduction(index):
    index_ = list()
    for i in range(len(index)):
        xiabiao = np.argmax(index[i], axis=1)
        out_index = ''
        for j in range(len(xiabiao)):
            if xiabiao[j] != 10:
                out_index = out_index + str(xiabiao[j])
        index_.append(int(out_index))
    return np.array(index_)


def loss_func(py_x, Y):
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 0], labels=Y[:, 0]))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 1], labels=Y[:, 1]))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 2], labels=Y[:, 2]))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 3], labels=Y[:, 3]))
    loss = tf.reduce_sum([loss0, loss1, loss2, loss3])
    return loss


def get_weights_bases(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weights, bases


def get_weights_bases_wout(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1], shape[2]], stddev=0.1))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[0], shape[2]]))
    return weights, bases


def model(X, w, w1, w2, w3, w4, w5, b5, w6, b6, p_keep_conv, p_keep_within):
    la = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l = tf.nn.avg_pool(la, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l = tf.nn.dropout(l, p_keep_conv)
    # shape = [?, 20, 30, 32]

    l1a = tf.nn.relu(tf.nn.conv2d(l, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.avg_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    # shape = [?, 10, 15, 64]

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.avg_pool(l2a, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_within)
    # shape = [?, 4, 5, 64]

    l3a_0 = tf.nn.relu(tf.nn.conv2d(l2, w3[0], strides=[1, 1, 1, 1], padding='SAME'))
    l3_0 = tf.nn.avg_pool(l3a_0, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_0 = tf.nn.dropout(l3_0, p_keep_within)
    l3a_1 = tf.nn.relu(tf.nn.conv2d(l2, w3[1], strides=[1, 1, 1, 1], padding='SAME'))
    l3_1 = tf.nn.avg_pool(l3a_1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_1 = tf.nn.dropout(l3_1, p_keep_within)
    l3a_2 = tf.nn.relu(tf.nn.conv2d(l2, w3[2], strides=[1, 1, 1, 1], padding='SAME'))
    l3_2 = tf.nn.avg_pool(l3a_2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_2 = tf.nn.dropout(l3_2, p_keep_within)
    l3a_2 = tf.nn.relu(tf.nn.conv2d(l2, w3[2], strides=[1, 1, 1, 1], padding='SAME'))
    l3_2 = tf.nn.avg_pool(l3a_2, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_2 = tf.nn.dropout(l3_2, p_keep_within)
    l3a_3 = tf.nn.relu(tf.nn.conv2d(l2, w3[3], strides=[1, 1, 1, 1], padding='SAME'))
    l3_3 = tf.nn.avg_pool(l3a_3, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    l3_3 = tf.nn.dropout(l3_3, p_keep_within)
    # shape = [?, 4, 5, 64]

    l4a_0 = tf.nn.relu(tf.nn.conv2d(l3_0, w4[0], strides=[1, 1, 1, 1], padding='SAME'))
    l4_0 = tf.reshape(l4a_0, [-1, w5.get_shape().as_list()[1]])
    l4_0 = tf.nn.dropout(l4_0, p_keep_within)
    l4a_1 = tf.nn.relu(tf.nn.conv2d(l3_1, w4[1], strides=[1, 1, 1, 1], padding='SAME'))
    l4_1 = tf.reshape(l4a_1, [-1, w5.get_shape().as_list()[1]])
    l4_1 = tf.nn.dropout(l4_1, p_keep_within)
    l4a_2 = tf.nn.relu(tf.nn.conv2d(l3_2, w4[2], strides=[1, 1, 1, 1], padding='SAME'))
    l4_2 = tf.reshape(l4a_2, [-1, w5.get_shape().as_list()[1]])
    l4_2 = tf.nn.dropout(l4_2, p_keep_within)
    l4a_3 = tf.nn.relu(tf.nn.conv2d(l3_3, w4[3], strides=[1, 1, 1, 1], padding='SAME'))
    l4_3 = tf.reshape(l4a_3, [-1, w5.get_shape().as_list()[1]])
    l4_3 = tf.nn.dropout(l4_3, p_keep_within)
    # shape = [?, 4, 5, 64]

    l5_0 = tf.nn.relu(tf.matmul(l4_0, w5[0]) + b5[0])
    l5_0 = tf.nn.dropout(l5_0, p_keep_conv)
    l5_1 = tf.nn.relu(tf.matmul(l4_1, w5[1]) + b5[1])
    l5_1 = tf.nn.dropout(l5_1, p_keep_conv)
    l5_2 = tf.nn.relu(tf.matmul(l4_2, w5[2]) + b5[2])
    l5_2 = tf.nn.dropout(l5_2, p_keep_conv)
    l5_3 = tf.nn.relu(tf.matmul(l4_3, w5[3]) + b5[3])
    l5_3 = tf.nn.dropout(l5_3, p_keep_conv)

    layer_0 = tf.matmul(l5_0, w6[0]) + b6[0]
    layer_1 = tf.matmul(l5_1, w6[1]) + b6[1]
    layer_2 = tf.matmul(l5_2, w6[2]) + b6[2]
    layer_3 = tf.matmul(l5_3, w6[3]) + b6[3]

    layer = tf.concat([layer_0, layer_1, layer_2, layer_3], 1)
    return tf.reshape(layer, shape=(-1, 4, 11))


def main():
    with tf.Graph().as_default() as graph:
        trX, trY_ = rd.get_data_train()
        tvX, tvY_ = rd.get_data_validation()
        teX, teY_ = rd.get_data_test()
        tf.reshape(trX, [-1, 40, 50, 3])
        tf.reshape(tvX, [-1, 40, 50, 3])
        tf.reshape(teX, [-1, 40, 50, 3])
        X = tf.placeholder("float", [None, 40, 50, 3], name='x-input')
        Y = tf.placeholder("float", [None, 4, 11], name='y-input')
        w = init_weight([3, 3, 3, 32])
        para_num = tf.size(w)
        w1 = init_weight([3, 3, 32, 64])
        para_num = tf.add(para_num, tf.size(w1))
        w2 = init_weight([3, 3, 64, 64])
        para_num = tf.add(para_num, tf.size(w2))
        w3 = init_weight([4, 3, 3, 64, 64])
        para_num = tf.add(para_num, tf.size(w3))
        w4 = init_weight([4, 3, 3, 64, 64])
        para_num = tf.add(para_num, tf.size(w3))
        w5, b5 = get_weights_bases_wout([4, 4*5*64, 400])
        para_num = tf.add(para_num, tf.size(w5))
        para_num = tf.add(para_num, tf.size(b5))
        w6, b6 = get_weights_bases_wout([4, 400, 11])
        para_num = tf.add(para_num, tf.size(w6))
        para_num = tf.add(para_num, tf.size(b6))
        p_keep_conv = tf.placeholder("float", name='p_keep_conv')
        p_keep_within = tf.placeholder("float", name='P_keep_within')
        py_x = model(X, w, w1, w2, w3, w4, w5, b5, w6, b6, p_keep_conv, p_keep_within)

        cost = loss_func(py_x, Y)
        train_scalar = tf.summary.scalar("value_train", cost)
        val_scalar = tf.summary.scalar("value_val", cost)
        train_op = tf.train.AdamOptimizer().minimize(cost)
        predict_op = py_x
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        # saver.restore(sess, './model/model.cpkt-98')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        trX, trY_r, tvX, tvY_v, teX, teY_e = sess.run([trX, trY_, tvX, tvY_, teX, teY_])
        trY = y_handle(trY_r)
        tvY = y_handle(tvY_v)
        teY = y_handle(teY_e)
        tv_jiance = y_reduction(tvY)
        te_jiance = y_reduction(teY)
        mac_acc = 0.0
        cost_min = 100.0
        print(sess.run(para_num))
        # para_num = tf.size([w, w1, w2, w3, w4, w5, b5, w6, b6])
        # print(tvY_jiance)

        for i in range(500):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                train_op_, cost_train, train_scalar_ = sess.run(
                    [train_op, cost, train_scalar],
                    feed_dict={X: trX[start:end], Y: trY[start:end],
                               p_keep_conv: 0.8, p_keep_within: 0.5})
            writer.add_summary(train_scalar_, i)
            yuce_Y, cost_val, val_scalar_ = sess.run(
                [predict_op, cost, val_scalar],
                feed_dict={X: tvX, Y: tvY, p_keep_conv: 1.0,
                           p_keep_within: 1.0})
            writer.add_summary(val_scalar_, i)
            yc_Y = y_reduction(yuce_Y)
            accuracy_val = np.mean(tv_jiance == yc_Y)
            print(i, accuracy_val, cost_val)
            if accuracy_val > mac_acc:
                mac_acc = accuracy_val
                cost_min = cost_val
                saver.save(sess, "./model/model.cpkt", global_step=i)
            elif accuracy_val == mac_acc:
                if cost_val < cost_min:
                    cost_min = cost_val
                    saver.save(sess, "./model/model.cpkt", global_step=i)
        yuce_Y_e = sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0, p_keep_within: 1.0})
        yc_Y_e = y_reduction(yuce_Y_e)
        accuracy_test = np.mean(te_jiance == yc_Y_e)
        print(np.mean([accuracy_val, accuracy_test]))
        writer.close()
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()



