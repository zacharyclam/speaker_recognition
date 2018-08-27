#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : enrollment.py
# @Time     : 2018/8/22 14:27 
# @Software : PyCharm
import os
from keras.models import load_model
from tqdm import tqdm
from absl import flags, app

try:
    import sys
    # 防止通过脚本运行时由于路径问题出现 ModuleNotFoundError
    sys.path.append(os.path.join(os.getcwd(), "code"))
    from utils.csv_util import features2csv
except ModuleNotFoundError:
    from code.utils.csv_util import features2csv

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", os.path.join(parent_dir, "data/enrollment_evalution"),
    "the enrolled data dir")

flags.DEFINE_string(
    "save_dir", os.path.join(parent_dir, "results/features"),
    "the save data dir")

flags.DEFINE_string(
    "weight_path", "D:\PythonProject\speakerRecognition\model\spk-01000-1.00.h5",
    "the model dir")

flags.DEFINE_string(
    "category", "test", "the category of data")

flags.DEFINE_integer(
    "enroll_sentence_nums", 20,
    "the enroll sentence nums")

flags.DEFINE_integer(
    "val_sentence_nums", 100,
    "the validate sentence nums")


def split_sentences(data_dir, save_dir, usage, enroll_sentence_nums=20, val_sentence_nums=100):
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
    enroll_list = []
    val_list = []

    for i, file_list in enumerate(data_list):
        enroll_list.append((file_list[:enroll_sentence_nums], i))
        val_list.append((file_list[enroll_sentence_nums:enroll_sentence_nums + val_sentence_nums], i))

    with open(os.path.join(save_dir, "enroll_list.txt"), "w") as f:
        for (file_list, label) in tqdm(enroll_list):
            for file in file_list:
                line = file + " " + str(label).zfill(4) + "\n"
                f.write(line)

    with open(os.path.join(save_dir, "validate_list.txt"), "w") as f:
        for (file_list, label) in tqdm(val_list):
            for file in file_list:
                line = file + " " + str(label).zfill(4) + "\n"
                f.write(line)


def main(argv):
    model = load_model(FLAGS.weight_path)
    # 分割 注册人 数据集 并写入txt
    split_sentences(FLAGS.data_dir, FLAGS.save_dir, FLAGS.category, enroll_sentence_nums=FLAGS.enroll_sentence_nums,
                    val_sentence_nums=FLAGS.val_sentence_nums)

    # 将注册人的注册语句特征写入csv文件
    features2csv(FLAGS.save_dir, category="enroll", model=model, mean=True, sentence_nums=FLAGS.enroll_sentence_nums)

    # 将注册人的验证语句特征写入csv文件
    features2csv(FLAGS.save_dir, category="validate", model=model, mean=False, sentence_nums=FLAGS.val_sentence_nums)


if __name__ == "__main__":
    app.run(main)
