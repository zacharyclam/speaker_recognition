#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : testRecord.py
# @Time     : 2018/8/20 21:10 
# @Software : PyCharm
import numpy as np
import os
from keras.models import load_model

from get_log_fbank import get_log_fbank


def getFeatures(model, dataListDir, mean=False, sentence_nums=20):
    def caculateFeatures(model, fb_input, mean=True):
        features = model.predict(fb_input)
        features = np.array(features)
        if mean:
            # (1,256)
            return np.mean(features, axis=0)
        else:
            # (N,256)
            return features

    fbank = get_log_fbank(dataListDir).reshape((11960, 1))
    # fb_input.append(fbank)
    fb_input = np.array(fbank)[np.newaxis, :, :]

    fearures = caculateFeatures(model, fb_input, mean)
    return fearures


def getCDS(a, b):
    """
    返回归一化后的余弦距离，得分CDS越接近1越好
    :param a: shape[1,-1]
    :param b: shape[1, -1]
    :return:
    """

    num = float(a.dot(b.T))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cds = num / denom  # 余弦值
    cds = 0.5 + 0.5 * cds  # 归一化
    return cds


model = load_model("spk.h5")

people1 = getFeatures(model, "a.wav")

people2 = getFeatures(model, "b.wav")

people3 = getFeatures(model, "c.wav")

print(getCDS(people1, people2))

print(getCDS(people1, people3))
print(getCDS(people2, people3))
