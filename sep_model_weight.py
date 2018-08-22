#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : calculate_score.py
# @Time     : 2018/8/17 14:11 
# @Software : PyCharm
from keras import Model
from keras.models import load_model


def sepModelWeight(weight_path, save_path):
    model = load_model(weight_path)
    seq = model.get_layer("sequential_1")
    targetModel = Model(inputs=model.input, outputs=seq.get_output_at(1))
    targetModel.save(save_path)


weight_path = "checkpoint-00356-0.98.h5"

sepModelWeight(weight_path, "spk-pool.h5")
