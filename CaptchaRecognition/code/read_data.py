# coding: utf-8

import tensorflow as tf

train_size = 32000
validation_size = 4000
test_size = 4000


def read_data(filename_queue):
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
    image = tf.reshape(image, [40, 50, 3])
    image = tf.cast(image, tf.float32) / tf.constant(255.)
    return image, label


def get_data_train(batch=train_size):
    filename_queue = tf.train.string_input_producer([
        '../data/tfrecords/train_%d.tfrecords' % i for i in range(0, 8)], num_epochs=1)
    image, label = read_data(filename_queue)
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=100)
    return out_image, out_label


def get_data_validation(batch=validation_size):
    filename_queue = tf.train.string_input_producer([
        '../data/tfrecords/validation_%d.tfrecords' % i for i in range(0, 1)], num_epochs=1)
    image, label = read_data(filename_queue)
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=100)
    return out_image, out_label


def get_data_test(batch=test_size):
    filename_queue = tf.train.string_input_producer([
        '../data/tfrecords/test_%d.tfrecords' % i for i in range(0, 1)], num_epochs=1)
    image, label = read_data(filename_queue)
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=100)
    return out_image, out_label
