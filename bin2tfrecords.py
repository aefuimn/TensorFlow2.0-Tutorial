# !/usr/bin/python3
# -*-coding:utf-8-*-
import os
import tensorflow as tf
import IPython.display as display
import numpy as np
from tqdm import tqdm
from time import sleep
"""
目录结构
bin
--data_batch_1.bin
--data_batch_2.bin
--data_batch_3.bin
--data_batch_4.bin
--data_batch_5.bin
"""
# 数据集目录
data_dir = "bin"
# 数据集文件列表
file_list = os.listdir(data_dir)
# 拼接目录
for i in range(len(file_list)):
    file_list[i] = os.path.join(data_dir, file_list[i])

# tfrecords文件名称
tfrecords = "test_cifar10.tfrecords"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_to_tfrecords():
    # tfrecords文件写入句柄
    writer = tf.io.TFRecordWriter(tfrecords)

    # 遍历所有二进制文件，将图片写入tfrecords文件中
    with tqdm(total=50000) as t:
        for i in range(5):
            # 文件名称
            file_name = file_list[i]
            file = open(file_name, 'rb').read()
            for j in range(10000):
                # 标签
                label = file[j * 3073]
                # 图像数据
                img_raw = file[(j * 3073) + 1: (j + 1) * 3073]
                # 转换为tensor
                img_tensor = tf.io.decode_raw(img_raw, tf.dtypes.uint8)
                # 转换形状
                img_tensor = tf.reshape(img_tensor, [32, 32, 3])
                # 将tensor序列化
                img_ser = tf.io.serialize_tensor(img_tensor)
                # 构建example
                feature = {
                    'label': _int64_feature(label),
                    'image_raw': _bytes_feature(img_ser)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # 将example写入tfrecords文件中
                writer.write(example.SerializeToString())
                t.update(1)
    writer.close()


def read_from_tfrecords():
    # 读取tfrecord文件
    dataset = tf.data.TFRecordDataset(tfrecords)
    # tfrecords格式文件
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = dataset.map(_parse_image_function)

    for img_features in parsed_image_dataset:
        img = img_features['image_raw'].numpy()
        display.display(display.Image(data=img))
        sleep(10)


if __name__ == '__main__':
    write_to_tfrecords()
