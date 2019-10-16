# -*-coding: utf-8-*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb

print(tf.__version__)


def load_imdb():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=256)
    return train_data, train_labels, test_data, test_labels


def build_model():
    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # 加载数据
    train_data, train_labels, test_data, test_labels = load_imdb()

    # 输出目录
    OUTPUT_PATH = '/home/output'
    # 批次大小
    BATCH_SIZE = 64
    # 迭代次数
    EPOCHS = 10

    # 构建模型
    model = build_model()
    model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # 评估模型
    model.evaluate(test_data, test_labels, verbose=2)

    # 保存模型
    model.save(OUTPUT_PATH)
