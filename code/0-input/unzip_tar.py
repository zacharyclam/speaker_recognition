#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : unzip_tar.py
# @Time     : 2018/8/17 9:01 
# @Software : PyCharm
import tarfile
import os
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", "../../data/data_aishell/wav/",
    "the orient tar data dir")

flags.DEFINE_string(
    "save_dir", "../../untar_data/",
    "the save untar data dir")


# 解压tar.gz文件到文件夹
def untar_dir(srcname, data_path):
    tar_handle = tarfile.open(srcname, "r:gz")
    tar_handle.extractall(data_path)
    tar_handle.close()


def untar(data_dir, save_dir):
    """
    
    @param data_dir:
    @param save_dir:
    @return:
    """
    # 将音频文件解压
    for root, dir, file in os.walk(data_dir):
        for filename in file:
            untar_dir(os.path.join(root, filename), save_dir)


def main(argv):
    untar(FLAGS.data_dir, FLAGS.save_dir)
    print("finished")


if __name__ == "__main__":
    app.run(main)
