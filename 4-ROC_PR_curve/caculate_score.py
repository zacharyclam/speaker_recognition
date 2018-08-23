#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : caculate_score.py
# @Time     : 2018/8/22 17:50 
# @Software : PyCharm

import tensorflow as tf
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

parent_dir =  os.path.abspath(os.path.join(os.getcwd(),".."))

tf.app.flags.DEFINE_string(
    "enrollment_dir", default=os.path.join(parent_dir,"2-enrollment"), help= "the dir of enrolllment"
)
tf.app.flags.DEFINE_string(
    "evalution_dir", default=os.path.join(parent_dir,"3-evalution"), help= "the dir of evalution"
)

FLAGS = tf.app.flags.FLAGS


def read_features(csv_dir, category):
    csv_path = os.path.join(csv_dir, category + "_features.csv")
    data = pd.read_csv(csv_path, encoding="utf-8")
    for label, features in data.values:
        yield label, np.array(list(map(float, features.split(","))))
    # for key, val in data_dict.items():
    #     print(key,val)
    #     data_dict[key] = list(map(float, val.split(",")))
    # return data_dict


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


if __name__ == "__main__":
    validate_dict = read_features(FLAGS.enrollment_dir, "validate")
    strange_dict = read_features(FLAGS.evalution_dir, "stranger")

    score_path = "score.txt"

    # score n/tp/fp
    with open(score_path, "w") as f:
        for val_label, val_feat in tqdm(validate_dict):
            enroll_dict = read_features(FLAGS.enrollment_dir, "enroll")
            distance = [getCDS(val_feat, enroll_feat) for _, enroll_feat in enroll_dict]
            predict_label = np.argmax(distance, axis=0)
            line = str(distance[predict_label]) + " "
            if predict_label == val_label:
                line += "tp\n"
            else:
                line += "fp\n"
            f.write(line)

        for stranger_label, stranger_feat in tqdm(strange_dict):
            enroll_dict = read_features(FLAGS.enrollment_dir, "enroll")
            distance = [getCDS(stranger_feat, enroll_feat) for _, enroll_feat in enroll_dict]
            predict_label = np.argmax(distance, axis=0)
            line = str(distance[predict_label]) + " n\n"
            f.write(line)

    # print("label:{} predict:{} score:{}".format(val_label, predict_label, distance[predict_label]))
