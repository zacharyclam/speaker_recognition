#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : data_feeder.py
# @Time     : 2018/8/24 17:45 
# @Software : PyCharm
# 生成数据
import numpy as np
from keras.utils import to_categorical


def generate_fit(path, batch_size, classe_nums):
    """

    :param path: 数据路径
    :param batch_size:
    :param classe_nums: 类别
    :return:
    """
    with open(path) as f:
        data_list = [line.strip().split(' ') for line in f]
    index = np.arange(len(data_list))
    while True:
        cnt = 0
        X = []
        Y = []
        # shffle data
        np.random.shuffle(index)
        data_list = np.array(data_list)[index, :]
        for bin_path, label in data_list:
            x = np.fromfile(bin_path, dtype=np.float).reshape((299, 40))
            # x = np.fromfile(bin_path, dtype=np.float)
            X.append(x)
            Y.append(label)

            cnt += 1

            if cnt % batch_size == 0:
                yield [np.array(X), np.array(to_categorical(Y, classe_nums))]
                X = []
                Y = []
                cnt = 0
