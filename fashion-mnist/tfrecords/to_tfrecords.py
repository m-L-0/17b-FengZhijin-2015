from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    with tf.python_io.TFRecordWriter('./data/tfrecords/'+name+'.tfrecords') as writer:
        print('Writing:  '+name+'.tfrecords')
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    data_file = './data/fashion/'
    data_sets = mnist.read_data_sets(data_file,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=5000)

    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')
