# coding: utf-8
# author: Fengzhijin
# time: 2017.12.5
# ==================================
'''
车牌字符识别数据集tfrecords文件读取
1.next_batch() - 读取指定文件和指定大小的数据
2.get_train() - 读取训练集数据
3.get_validation() - 读取验证集数据
4.get_test() - 读取测试集数据
'''

import tensorflow as tf
import numpy as np


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


def next_batch(classes, batch=validation_size):
    if classes == 'train':
        filename_queue = tf.train.string_input_producer(
            ["../data/tfrecords/train.tfrecords"], num_epochs=1)
    else:
        filename_queue = tf.train.string_input_producer(
            ["../data/tfrecords/validation.tfrecords"], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label']
    image = tf.reshape(image, [48, 24, 3])
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=3 * batch,
                                                  min_after_dequeue=batch+100)
    out_image = tf.cast(out_image, tf.float32) / tf.constant(255.)

    with tf.Session() as sess:
        init_op = [tf.global_variables_initializer(),
                   tf.local_variables_initializer()]
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images, labels = sess.run([out_image, out_label])
        coord.request_stop()
        coord.join(threads)
        la = np.zeros([batch, 66], dtype=int)
        for i in range(batch):
            la[i][labels[i]] = 1
        return images, la


def get_train(batch=train_size):
    filename_queue = tf.train.string_input_producer(
        ["../data/tfrecords/train.tfrecords"], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label']
    image = tf.reshape(image, [48, 24, 3])
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=batch+100)
    out_image = tf.cast(out_image, tf.float32) / tf.constant(255.)

    with tf.Session() as sess:
        init_op = [tf.global_variables_initializer(),
                   tf.local_variables_initializer()]
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images, labels = sess.run([out_image, out_label])
        coord.request_stop()
        coord.join(threads)
        la = np.zeros([batch, 66], dtype=int)
        for i in range(batch):
            la[i][labels[i]] = 1
        return images, la


def get_validation(batch=validation_size):
    filename_queue = tf.train.string_input_producer(
        ["../data/tfrecords/validation.tfrecords"], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label']
    image = tf.reshape(image, [48, 24, 3])
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=batch+100)
    out_image = tf.cast(out_image, tf.float32) / tf.constant(255.)

    with tf.Session() as sess:
        init_op = [tf.global_variables_initializer(),
                   tf.local_variables_initializer()]
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images, labels = sess.run([out_image, out_label])
        coord.request_stop()
        coord.join(threads)
        la = np.zeros([batch, 66], dtype=int)
        for i in range(batch):
            la[i][labels[i]] = 1
        return images, la


def get_test(batch=test_size):
    filename_queue = tf.train.string_input_producer(
        ["../data/tfrecords/test.tfrecords"], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label']
    image = tf.reshape(image, [48, 24, 3])
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=batch+100)
    out_image = tf.cast(out_image, tf.float32) / tf.constant(255.)

    with tf.Session() as sess:
        init_op = [tf.global_variables_initializer(),
                   tf.local_variables_initializer()]
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images, labels = sess.run([out_image, out_label])
        coord.request_stop()
        coord.join(threads)
        la = np.zeros([batch, 66], dtype=int)
        for i in range(batch):
            la[i][labels[i]] = 1
        return images, la
