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
from absl import flags
from absl import app


FLAGS = flags.FLAGS

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

flags.DEFINE_string(
    "data_dir", os.path.join(parent_dir, "data/enrollment_evalution"),
    "the enrolled data dir")

flags.DEFINE_string(
    "weight_path", "D:\PythonProject\speakerRecognition\model\spk-00152-0.88.h5",
    "the model dir")

flags.DEFINE_string(
    "category", "test", "the category of data")

flags.DEFINE_string(
    "save_dir", os.path.join(parent_dir, "3-evalution"),
    "the strangers' features save dir")

flags.DEFINE_integer(
    "stranger_sentence_nums", 100,
    "the stranger sentence nums")


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
            fbank = np.fromfile(bin_path, dtype=np.float)
            fbank_list.append(fbank)

            cnt += 1
            if cnt % sentence_nums == 0:
                features = caculate_features(np.array(fbank_list)[:, :, np.newaxis])
                cnt = 0
                fbank_list = []
                if mean is True:
                    people_list.append((label.rstrip("\n"), ",".join(str(feat) for feat in features)))
                else:
                    for feature in features:
                        people_list.append((label.rstrip("\n"), ",".join(str(feat) for feat in feature)))

    # 将特征写入 csv 文件
    features_df = pd.DataFrame(people_list, columns=["label", "features_str"])
    df_save_path = os.path.join(save_dir, category + "_features.csv")
    features_df.to_csv(df_save_path, index=False, encoding="utf-8")


def read_features(csv_dir, category):
    csv_path = os.path.join(csv_dir, category + "_features.csv")
    data = pd.read_csv(csv_path, encoding="utf-8")
    data_list = data.values
    for label, features in data_list:
        yield label, list(map(float, features.split(",")))


def main(argv):
    model = load_model(FLAGS.weight_path)

    # 分割 陌生人 数据集 并写入txt
    split_data(FLAGS.data_dir, FLAGS.save_dir, FLAGS.category, sentence_nums=FLAGS.stranger_sentence_nums)

    # 将陌生人的注册语句特征写入csv文件
    features2csv(FLAGS.save_dir, "stranger", model, mean=False,
                 sentence_nums=FLAGS.stranger_sentence_nums)

    # 读取陌生人特征信息
    # enroll_features = read_features(FLAGS.save_dir, "stranger")


if __name__ == "__main__":
    app.run(main)
