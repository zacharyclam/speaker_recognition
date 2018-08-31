#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tes.py
# @Time     : 2018/8/31 9:32 
# @Software : PyCharm
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxoutDense,  BatchNormalization, Conv1D, Flatten, MaxPool1D
from keras.utils import plot_model


def construct_model(classe_nums):
    model = Sequential()

    model.add(
        Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', input_shape=(99, 40), name='block1_conv1'))
    model.add(MaxPool1D(pool_size=2, name='block1_pool1'))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1))

    model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', name='block1_conv2'))
    model.add(MaxPool1D(pool_size=2, name='block1_pool2'))

    model.add(Flatten(name='block1_flat1'))
    model.add(Dropout(0.5, name='block1_drop1'))

    model.add(Dense(512, activation='relu', name='block2_dense2'))
    model.add(MaxoutDense(512, nb_feature=4, name="block2_maxout2"))
    model.add(Dropout(0.5, name='block2_drop2'))

    model.add(Dense(512, activation='relu', name='block2_dense3', kernel_regularizer=l2(1e-4)))
    model.add(MaxoutDense(512, nb_feature=4, name="block2_maxout3"))
    model.add(Dense(classe_nums, activation='softmax', name="predict"))

    plot_model(model, to_file='model_struct.png', show_shapes=True, show_layer_names=False)

    model.summary()


if __name__ == "__main__":
    construct_model(classe_nums=340)
