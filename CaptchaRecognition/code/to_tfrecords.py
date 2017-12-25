# coding: utf-8
# author: Fengzhijin
# time: 2017.12.12
# ==================================
'''
实现了将验证码数据集源文件转换成tfrecords文件
1._int64_feature() - int64数据转换函数
2._bytes_feature() - 二进制字符串转换函数
3.convert_to() - tfrecords文件生成函数
'''

from PIL import Image
import os
import tensorflow as tf
import csv

size = 40000


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to():
    cwd_images = "../data/captcha/images/"
    cwd_name = "data/captcha/images/"
    cwd_labels = "../data/captcha/labels/labels.csv"
    tf_file = "../data/tfrecords/"

    train_size = 0
    validation_size = 0
    test_size = 0
    num = 0
    labels = {}
    sizes = {}
    with open(cwd_labels) as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            if len(i[1]) not in sizes:
                sizes[len(i[1])] = 1
            else:
                sizes[len(i[1])] = sizes[len(i[1])] + 1
            labels[i[0]] = int(i[1])
    for img_name in os.listdir(cwd_images):
        num += 1
        img_path = cwd_images + img_name
        image = Image.open(img_path)
        image = image.resize((50, 40))
        image_raw = image.tobytes()
        index = labels[cwd_name + img_name]
        print(cwd_name + img_name + ":" + str(index))
        example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(int(index)),
                    'image_raw': _bytes_feature(image_raw)}))
        if num % 10 <= 8 and num % 10 > 0:
            train_size += 1
            if train_size % 4000 == 1:
                train_few = train_size // 4000
                writer_train = tf.python_io.TFRecordWriter(tf_file+'train_'+str(train_few)+'.tfrecords')
            writer_train.write(example.SerializeToString())
        elif num % 10 == 9:
            validation_size += 1
            if validation_size % 4000 == 1:
                validation_few = validation_size // 4000
                writer_validation = tf.python_io.TFRecordWriter(tf_file+'validation_'+str(validation_few)+'.tfrecords')
            writer_validation.write(example.SerializeToString())
        elif num % 10 == 0:
            test_size += 1
            if test_size % 4000 == 1:
                test_few = test_size // 4000
                writer_test = tf.python_io.TFRecordWriter(tf_file+'test_'+str(test_few)+'.tfrecords')
            writer_test.write(example.SerializeToString())

    print("num = %d" % num)
    print(sizes)
    writer_train.close()
    writer_validation.close()
    writer_test.close()


if __name__ == "__main__":
    convert_to()
