# -*- coding: utf-8 -*-
"""
TensorFlow2.0 kerasapi 分布式训练 Cifar10
训练环境：3个GPU 1080Ti + 2*1070Ti
训练集准确率：98.27%
验证集准确率：84.77%
测试机准确率：84.34%
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import tensorflow as tf
from tensorflow import keras

# TensorFlow日志显示等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 加载数据
def load_data():
    train_data = {b'data': [], b'labels': []}
    # 训练数据
    for i in range(5):
        with open("cifar-10-batches-py/data_batch_" + str(i+1), mode='rb') as f:
            data = pickle.load(f, encoding="bytes")
            train_data[b'data'] += list(data[b'data'])
            train_data[b'labels'] += data[b'labels']

    # 测试数据
    with open("cifar-10-batches-py/test_batch", mode='rb') as file:
        test_data = pickle.load(file, encoding='bytes')

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_data[b'data'][:45000], train_data[b'labels'][:45000]))
    train_ds = train_ds.map(image_enhancement, num_parallel_calls=10)
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).repeat()
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (train_data[b'data'][45000:], train_data[b'labels'][45000:]))
    validation_ds = validation_ds.map(image_enhancement, num_parallel_calls=10)
    validation_ds = validation_ds.batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data[b'data'], test_data[b'labels']))
    test_ds = test_ds.map(test_image, num_parallel_calls=10)
    test_ds = test_ds.batch(BATCH_SIZE)
    return train_ds, validation_ds, test_ds


# 训练图增强函数
def image_enhancement(image, label):
    image = tf.dtypes.cast(image, tf.dtypes.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    # 图片随机剪裁
    # image = tf.image.random_crop(image, [28, 28, 3])
    # 图片随机翻转
    image = tf.image.random_flip_left_right(image)
    # 图片随机调整亮度
    image = tf.image.random_brightness(image, max_delta=63)
    # 图片随机调整对比度
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # 归一化处理
    image = tf.image.per_image_standardization(image)
    return image, label


# 测试图片处理
def test_image(image, label):
    image = tf.dtypes.cast(image, tf.dtypes.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    # 图片剪裁
    # image = tf.image.resize(image, [28, 28])
    # 归一化处理
    image = tf.image.per_image_standardization(image)
    return image, label


# 搭建模型
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, 3, activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


# 学习率下降函数
def decay(epoch):
    if epoch < 30:
        return 1e-2
    elif epoch >= 30 and epoch < 50:
        return 5e-3
    elif epoch >= 50 and epoch < 80:
        return 1e-3
    elif epoch >= 80 and epoch < 120:
        return 1e-4
    else:
        return 1e-5


# 回调函数打印每个迭代的学习率
class PrintLR(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                          model.optimizer.lr.numpy()))


if __name__ == "__main__":
    # 每个副本批次大小
    BATCH_SIZE_PER_REPLICA = 256
    # 迭代次数
    EPOCHS = 160
    # 输出路径
    OUTPUT_PATH = "/home/output"

    # 定义checkpoint目录保存checkpoints
    checkpoint_dir = os.path.join(OUTPUT_PATH, 'train_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # keras回调函数
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_PATH, 'logs')),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                        save_weights_only=True),
        keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

    # 启动分布式训练策略
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # 搭建分布式模型
    with strategy.scope():
        model = build_model()

    # 分布式批次大小
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # 加载数据
    train_dataset, validation_dataset, test_dataset = load_data()

    # 训练模型
    model.fit(train_dataset,
              epochs=EPOCHS,
              callbacks=callbacks,
              verbose=2,
              steps_per_epoch=50000//BATCH_SIZE,
              validation_steps=6,
              validation_data=validation_dataset)

    # 评估模型
    model.evaluate(test_dataset)

    # 保存模型
    # 可以配置save_format参数，默认为'tf',模型文件保存为sava_model格式
    # 可配置为'h5'，模型将保存为单个HDF5文件
    model.save(os.path.join(OUTPUT_PATH, 'model'))
