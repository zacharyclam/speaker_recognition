#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : evalution.py
# @Time     : 2018/8/22 17:33 
# @Software : PyCharm
import os
import numpy as np
from keras.models import load_model
from tqdm import tqdm
import pandas as pd


def split_data(data_dir, save_dir, usage, sentence_nums=20):
    """

    :paramdataDir: 文件存储路径
    :param usage: 数据集类别
    :param enroll_num: 注册使用语句条数
    :param val_num: 验证使用语句条数
    :return:
    """
    data_path = os.path.join(data_dir, usage)
    # 获取文件列表
    data_list = []
    for dir in os.listdir(data_path):
        file_list = os.listdir(os.path.join(data_path, dir))
        data_list.append([os.path.join(data_path, dir, file) for file in file_list])

    # (list , label)
    stranger_list = []

    for i, file_list in enumerate(data_list):
        stranger_list.append((file_list[:sentence_nums], i))

    with open(os.path.join(save_dir, "stranger_list.txt"), "w") as f:
        for (file_list, label) in tqdm(stranger_list):
            for file in file_list:
                line = file + " " + str(label).zfill(4) + "\n"
                f.write(line)


def features2csv(data_list_dir, save_dir, category, model, mean=True, sentence_nums=20):
    def caculate_features(fb_input, mean=mean):
        features = model.predict(fb_input)
        features = np.array(features)
        if mean:
            # (1,256)
            return np.mean(features, axis=0)
        else:
            # (N,256)
            return features

    data_path = os.path.join(data_list_dir, category + "_list.txt")

    # (label, features)
    people_dict = []
    with open(data_path) as f:
        X = []
        cnt = 0
        for line in tqdm(f):
            bin_path, label = line.split(" ")
            x = np.fromfile(bin_path, dtype=np.float)
            X.append(x)

            cnt += 1
            if cnt % sentence_nums == 0:
                features = caculate_features(np.array(X)[:, :, np.newaxis], mean)
                cnt = 0
                X = []
                if mean is True:
                    people_dict.append((label.rstrip("\n"), ",".join(str(feat) for feat in features)))
                else:
                    label = label.rstrip("\n")
                    for feature in features:
                        people_dict.append((label.rstrip("\n"), ",".join(str(feat) for feat in feature)))

    features_df = pd.DataFrame(people_dict, columns=["label", "features_str"])
    df_save_path = os.path.join(save_dir, category + "_features.csv")
    features_df.to_csv(df_save_path, index=False, encoding="utf-8")


def read_features(csv_dir, category):
    csv_path = os.path.join(csv_dir, category + "_features.csv")
    data = pd.read_csv(csv_path, encoding="utf-8")
    data_dict = data.set_index("label").to_dict()["features_str"]

    for key,val in data_dict.items():
        data_dict[key] = list(map(float, val.split(",")))
    return data_dict


def getList(save_dir, category):
    """
    将bin文件路径及标签写入txt

    :param save_dir: 保存文件路径
    :param usage: 数据集类别
    :return:
    """
    tname = os.path.join(save_dir, category + "_list.txt")
    data_dir = os.path.join(save_dir, category)
    # 获取子文件夹下的文件列表
    sub_dir = os.listdir(data_dir)

    with open(tname, "w") as f:
        for subname in sub_dir:
            subpath = os.path.join(data_dir, subname)
            for filename in os.listdir(subpath):
                # 文件路径 标签
                line = os.path.join(subpath, filename) + " " + subname + "\n"
                f.write(line)


if __name__ == "__main__":
    # 获取上级目录
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = os.path.join(parent_dir, "data/enrollment_evalution")
    weight_path = "D:\PythonProject\speakerRecognition\spk_pool.h5"
    category = "test"
    save_dir = os.path.join(parent_dir,"3-evalution")
    stranger_sentence_nums = 100

    model = load_model(weight_path)

    # 分割 陌生人 数据集 并写入txt
    # split_data(data_dir, save_dir, category, sentence_nums=stranger_sentence_nums)

    # 将陌生人的注册语句特征写入csv文件
    features2csv(save_dir, save_dir, "stranger", model, mean=False, sentence_nums=stranger_sentence_nums)


    # 读取陌生人特征信息
    # enroll_features = read_features(save_dir,"stranger") 