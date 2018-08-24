#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : evalution.py
# @Time     : 2018/8/22 17:33 
# @Software : PyCharm
import os
from keras.models import load_model
from tqdm import tqdm
from absl import flags
from absl import app

import sys
sys.path.append(os.getcwd())
from ..utils.csv_util import features2csv

FLAGS = flags.FLAGS

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

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
