#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : model.py
# @Time     : 2018/8/24 14:49 
# @Software : PyCharm
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, MaxoutDense,  BatchNormalization, Conv1D, Flatten, MaxPool1D
from keras.models import Model


def construct_model(classe_nums):
    model = Sequential()

    model.add(
        Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', input_shape=(99, 40), name='block1_conv1'))
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
    model.add(Dropout(0.5, name='block2_drop3'))

    model.summary()

    model_input = Input(shape=(99, 40))
    features = model(model_input)
    extract_feature_model = Model(inputs=model_input, outputs=features)

    category_predict = Dense(classe_nums, activation='softmax', name="predict")(features)

    sr_model = Model(inputs=model_input, outputs=category_predict)

    return extract_feature_model, sr_model


def construct_model_dnn(classe_nums):
    model = Sequential()

    model.add(Dense(256, input_shape=(3960,), name="dense1"))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-3))
    model.add(MaxoutDense(256, nb_feature=4, name="maxout1"))

    model.add(Dense(256, name="dense2", activation='relu'))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-3))
    model.add(MaxoutDense(256, nb_feature=4, name="maxout2"))

    model.add(Dense(256, name="dense3", activation='relu'))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-3))
    model.add(MaxoutDense(256, nb_feature=4, name="maxout3"))
    model.add(Dropout(0.5, name="drop3"))

    model.add(Dense(256, name="dense4", activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-3))
    model.add(MaxoutDense(256, nb_feature=4, name="maxout4"))
    model.add(Dropout(0.5, name="drop4"))

    model.summary()

    model_input = Input(shape=(3960,))
    features = model(model_input)
    extract_feature_model = Model(inputs=model_input, outputs=features)
    category_predict = Dense(classe_nums, activation='softmax', name="predict")(features)
    sr_model = Model(inputs=model_input, outputs=category_predict)

    return extract_feature_model, sr_model


if __name__ == "__main__":
    extract_feature_model, sr_model = construct_model(classe_nums=340)
