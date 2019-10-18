# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import pickle
import tensorflow as tf

# 配置多Worker环境变量'TF_CONFIG'
# 'cluster': 集群host集合
# 'task': worker任务, index代表了是worker编号，worker0有更多工作需要作
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['10.10.2.37:12345', '10.10.2.38:12345']
    },
    'task': {'type': 'worker', 'index': 0}
})


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


# 模型类
class Alexnet(tf.keras.Model):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', name='conv3')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2, name='pool1')
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2, name='pool2')
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2, name='pool3')
        self.batch_normalization1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.batch_normalization2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.batch_normalization3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.flatten = tf.keras.layers.Flatten(name='f1')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.d1 = tf.keras.layers.Dense(1024, activation='relu', name='d1')
        self.d2 = tf.keras.layers.Dense(512, activation='relu', name='d2')
        self.d3 = tf.keras.layers.Dense(256, activation='relu', name='d3')
        self.d4 = tf.keras.layers.Dense(10, activation='softmax', name='out')

    def call(self, x):
        # conv1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch_normalization1(x)

        # conv2
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch_normalization2(x)

        # conv3
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch_normalization3(x)

        # Flatten
        x = self.flatten(x)
        x = self.dropout(x)

        # Dense
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


# 学习率衰减函数
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
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                          multi_worker_model.optimizer.lr.numpy()))


if __name__ == "__main__":
    # 每个worker的批次大小
    BATCH_SIZE_PER_WORKER = 384
    # worker数量
    NUM_WORKERS = 2
    # 总的批次大小
    BATCH_SIZE = BATCH_SIZE_PER_WORKER * NUM_WORKERS
    # 输出路径
    OUTPUT_PATH = '/home/output'
    # 迭代次数
    EPOCHS = 2

    # 定义checkpoint目录保存checkpoints
    checkpoint_dir = os.path.join(OUTPUT_PATH, 'train_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # keras回调函数
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_PATH, 'logs')),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

    # 启用多worker分布式训练策略
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # 数据集的创建，模型搭建和编译需要在'strategy.scope()'中完成
    with strategy.scope():
        train_dataset, validation_dataset, test_dataset = load_data()
        multi_worker_model = Alexnet()
        multi_worker_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                                   metrics=['accuracy'])

    # 训练模型
    multi_worker_model.fit(train_dataset,
                           epochs=EPOCHS,
                           validation_steps=6,
                           validation_data=validation_dataset,
                           steps_per_epoch=45000//BATCH_SIZE,
                           callbacks=callbacks,
                           verbose=2)

    # 评估模型
    multi_worker_model.evaluate(test_dataset)

    # 保存模型
    # 可以配置save_format参数，默认为'tf',模型文件保存为sava_model格式
    # 可配置为'h5'，模型将保存为单个HDF5文件
    multi_worker_model.save(os.path.join(OUTPUT_PATH, 'model'))
