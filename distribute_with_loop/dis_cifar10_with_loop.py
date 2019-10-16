# -*- coding: utf-8 -*-
"""
TensorFlow2.0 传统循环 分布式训练 Cifar10
训练环境：3个GPU 1080Ti + 2*1070Ti
训练集准确率：99.22%
验证集准确率：85.52%
测试机准确率：85.23%
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import pickle
import tensorflow as tf


# 启用分布式训练策略
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


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
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)
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
        self.conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2)
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2)
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2)
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(512, activation='relu')
        self.d3 = tf.keras.layers.Dense(256, activation='relu')
        self.d4 = tf.keras.layers.Dense(10, activation='softmax')

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
def decay_lr(epoch):
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


# 计算Loss函数
def compute_loss(labels, predictions):
    per_replica_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_replica_loss, global_batch_size=BATCH_SIZE)


# 单副本训练函数
def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss


# 单副本测试函数
def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


# 分布式训练函数
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)

# 分布式测试函数
@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


if __name__ == "__main__":

    # 每个设备的批次大小
    BATCH_SIZE_PER_DEVICE = 128
    # 迭代次数
    EPOCHS = 160
    # 输出路径
    OUTPUT_PATH = '/home/output'

    # 总批次大小
    BATCH_SIZE = BATCH_SIZE_PER_DEVICE * strategy.num_replicas_in_sync

    # 搭建分布式模型，需要分布式训练，模型和优化器必须在strategy.scope()下创建
    with strategy.scope():
        model = Alexnet()
        optimizer = tf.keras.optimizers.Adam(1e-2)

    # 定义训练Loss对象
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    # 定义测试loss，训练和测试的acc
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    # 加载数据
    train_dataset, validation_dataset, test_dataset = load_data()

    # 训练过程中输出模板
    template = ("\nEpoch {}/{}-{:.2f}s \nTrain Loss: {:.2f}, Train Accuracy: {:.2f}%"
                "\nVal Loss: {:.2f}, Val Accuracy: {:.2f}%")

    # 进行迭代
    for epoch in range(EPOCHS):
        # 循环训练
        tatal_loss = 0.0  # 总损失
        num_batches = 0  # 总批次
        # 迭代开始时间
        epoch_start_time = time.time()

        # 设置学习率
        optimizer.lr.assign(decay_lr(epoch))

        for x in train_dataset:
            tatal_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = tatal_loss / num_batches

        # 循环测试
        for x in validation_dataset:
            distributed_test_step(x)

        # 迭代结束时间
        epoch_end_time = time.time()

        # 打印迭代信息

        print(template.format(epoch + 1, EPOCHS,
                              epoch_end_time - epoch_start_time,
                              train_loss,
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        print('Learning rate for epoch {} is {}'.format(epoch + 1,
                                                        optimizer.lr.numpy()))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

        # 如果是最后一次迭代，进行测试
        if epoch + 1 == EPOCHS:
            for x in test_dataset:
                distributed_test_step(x)
            # 打印测试结果
            print('\nTest Loss: {:.2f}, Test Accuracy: {:.2f}%'.format(test_loss.result(), test_accuracy.result() * 100))

    # 保存模型
    model.save(os.path.join(OUTPUT_PATH, 'model'))
