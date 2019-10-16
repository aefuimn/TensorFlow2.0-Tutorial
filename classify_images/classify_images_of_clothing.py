# -*-coding: utf-8-*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


# 加载官方数据，加载速度会受网络环境影响
def load_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Explore数据
    print('train_images_shape:', train_images.shape)
    print('train_labels:', train_labels)
    print('train_labels_length:', len(train_labels))
    print('test_images_shape:', test_images.shape)
    print('test_images_length:', len(test_labels))

    # 归一化
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def build_model():
    # 使用TensorFlow2.0 提供的KerasAPI构建模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # 数据标签种类
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # 模型保存路径
    OUTPUT_PATH = '/home/output'
    # 批次大小
    BATCH_SIZE = 64
    # 迭代次数
    EPOCHS = 10

    # 加载数据
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()

    # 搭建模型
    model = build_model()

    # 训练
    model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\n Test accuracy:', test_acc)

    # 保存模型
    model.save(OUTPUT_PATH)
