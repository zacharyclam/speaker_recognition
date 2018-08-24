#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tvlistProduce.py
# @Time     : 2018/8/12 12:09 
# @Software : PyCharm

import os
import argparse
import sys


def getList(saveDir, category):
    """
    将bin文件路径及标签写入txt

    :param saveDir: 保存文件路径
    :param usage: 数据集类别
    :return:
    """
    tname = os.path.join(saveDir, category + "_list.txt")
    dataDir = os.path.join(saveDir, category)
    # 获取子文件夹下的文件列表
    subDir = os.listdir(dataDir)

    with open(tname, "w") as f:
        for i, subname in enumerate(subDir):
            subpath = os.path.join(dataDir, subname)
            for filename in os.listdir(subpath):
                # 文件路径 标签
                line = os.path.join(subpath, filename) + " " + str(i) + "\n"
                f.write(line)


parse = argparse.ArgumentParser()

parse.add_argument("--save_dir", type=str, help="save list to dir")
parse.add_argument("--category", type=str)

if __name__ == "__main__":
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    saveDir = flags.save_dir
    category = flags.category

    getList(saveDir, category)
    # python3 get_data_list.py --save_dir="data/bin/" --category="train"

    # python get_data_list.py --save_dir="data/bin/" --category="test"
    # python get_data_list.py --save_dir="data/bin/" --category="train"
