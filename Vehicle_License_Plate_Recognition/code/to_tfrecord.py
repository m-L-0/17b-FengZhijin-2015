# coding: utf-8
# author: Fengzhijin
# time: 2017.12.5
# ==================================
'''
实现了将车牌字符识别数据集源文件转换成tfrecords文件
1._int64_feature() - int64数据转换函数
2._bytes_feature() - 二进制字符串转换函数
3.convert_to() - tfrecords文件生成函数
'''

from PIL import Image
import os
import tensorflow as tf


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
validation_size = 630
train_size = 17819
size = 18449


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))


def convert_to():
    cwd = '../测试图像集/'
    # writer_train = tf.python_io.TFRecordWriter('../data/tfrecords/train.tfrecords')
    # writer_validation = tf.python_io.TFRecordWriter('../data/tfrecords/validation.tfrecords')
    writer_test = tf.python_io.TFRecordWriter('../data/tfrecords/test.tfrecords')
    sum = 0
    for i in classes:
        if i == '字母':
            classes_1 = classes_zimu
        elif i == '汉字':
            classes_1 = classes_hanzi
        elif i == '数字':
            classes_1 = classes_shuzi
        for index in classes_1:
            class_path = cwd + i + '/' + classes_1[index] + '/'
            print("写入"+classes_1[index]+"数据")
            # sum_validation = 0
            # sum_train = 0
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                image = Image.open(img_path)
                image = image.resize((24, 48))
                image_raw = image.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(int(index)),
                    'image_raw': _bytes_feature(image_raw)}))
                # if (sum < 10):
                #     sum_validation += 1
                #     writer_validation.write(example.SerializeToString())
                # elif (sum >= 10):
                #     sum_train += 1
                #     writer_train.write(example.SerializeToString())
                writer_test.write(example.SerializeToString())
                sum += 1
            # print("sum_validation = %d" % sum_validation)
            # print("sum_train = %d" % sum_train)
    # writer_train.close()
    # writer_validation.close()
    print("sum = %d" % sum)
    writer_test.close()


if __name__ == "__main__":
    convert_to()
