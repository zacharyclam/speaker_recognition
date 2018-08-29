#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tvlistProduce.py
# @Time     : 2018/8/12 12:09 
# @Software : PyCharm
import os
from absl import flags, app


def get_list(save_dir, category):
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
        for i, subname in enumerate(sub_dir):
            subpath = os.path.join(data_dir, subname)
            for filename in os.listdir(subpath):
                # 文件路径 标签
                line = os.path.join(subpath, filename) + " " + str(i) + "\n"
                f.write(line)


root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
FLAGS = flags.FLAGS

flags.DEFINE_string("save_dir", os.path.join(root_dir, "data/bin/"), "save list to dir")
flags.DEFINE_string("category", "validate", "the category of data")


def main(argv):
    get_list(os.path.abspath(FLAGS.save_dir), FLAGS.category)


if __name__ == "__main__":
    app.run(main)
    # usage
    # python3 get_data_list.py --save_dir="../../data/bin/" --category="validate"
    # python3 get_data_list.py --save_dir="../../data/bin/" --category="train"
