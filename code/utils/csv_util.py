#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : csv_util.py
# @Time     : 2018/8/24 14:20 
# @Software : PyCharm
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def features2csv(save_dir, category, model, mean=True, sentence_nums=20):
    def caculate_features(fb_input):
        """

        :param fb_input: fbank特征向量
        :return:  d-vector
        """
        features = model.predict(fb_input)
        features = np.array(features)
        if mean:
            # (1,256)
            return np.mean(features, axis=0)
        else:
            # (N,256)
            return features

    data_path = os.path.join(save_dir, category + "_list.txt")

    # (label, features)
    people_list = []
    with open(data_path) as f:
        fbank_list = []
        cnt = 0
        for line in tqdm(f):
            bin_path, label = line.split(" ")
            fbank = np.fromfile(bin_path, dtype=np.float).reshape((299, 40))
            # fbank = np.fromfile(bin_path, dtype=np.float)
            fbank_list.append(fbank)

            cnt += 1
            if cnt % sentence_nums == 0:
                features = caculate_features(np.array(fbank_list))
                cnt = 0
                fbank_list = []
                if mean is True:
                    people_list.append((label.rstrip("\n"), ",".join(str(feat) for feat in features)))
                else:
                    # 不对特征向量求均值
                    for feature in features:
                        people_list.append((label.rstrip("\n"), ",".join(str(feat) for feat in feature)))

    # 将特征写入 csv 文件
    features_df = pd.DataFrame(people_list, columns=["label", "features_str"])
    df_save_path = os.path.join(save_dir, category + "_features.csv")
    features_df.to_csv(df_save_path, index=False, encoding="utf-8")


def read_features(csv_dir, category):
    """

    :param csv_dir:
    :param category:
    :return:  label, features array
    """
    csv_path = os.path.join(csv_dir, category + "_features.csv")
    # 读取文件
    data = pd.read_csv(csv_path, encoding="utf-8")
    for label, features in data.values:
        yield label, np.array(list(map(float, features.split(","))))
