#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : process_data.py
# @Time     : 2018/8/11 18:02 
# @Software : PyCharm
import os
import re
from tqdm import tqdm
from utils.get_log_fbank import get_log_fbank
import argparse
import sys


def split_data(data_dir, usage):

    data_path = os.path.join(data_dir, usage)

    # 获取文件列表
    data_list = []
    for idx, dir in enumerate(os.listdir(data_path)):
        file_list = os.listdir(os.path.join(data_path, dir))
        data_list.append(([os.path.join(data_path, dir, file) for file in file_list], idx))

    return data_list


def wav2fb(data_dir, save_dir, usage):
    """
    将wav音频文件提取 logfbank 特征后写入.bin二进制文件
    :param data_dir: wav文件路径
    :param save_dir: 保存.bin文件路径
    :param usage: 数据集类别
    :return:
    """
    save_path = save_dir

    data_list = split_data(data_dir, usage)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fileList, label in tqdm(data_list):
        # 创建子文件夹
        sub_dir = os.path.join(save_path, category, str(label).zfill(4))
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        for wpath in fileList:
            fbank_feat = get_log_fbank(wpath)
            if fbank_feat is not None:
                file_name = re.search(r"B\S+", wpath).group(0)[:-4]
                fbank_feat.tofile(os.path.join(sub_dir, file_name + ".bin"))


parse = argparse.ArgumentParser()

parse.add_argument("--data_dir", type=str, default="D:\PythonProject\speakerRecognition\data")
parse.add_argument("--save_dir", type=str, default="D:\PythonProject\speakerRecognition\data\enrollment_evalution")
parse.add_argument("--category", type=str, help="the category of data", default="test")

if __name__ == "__main__":
    flags, unparsed = parse.parse_known_args(sys.argv[1:])
    # 解析命令行参数
    dataDir = flags.data_dir
    saveDir = flags.save_dir
    category = flags.category

    # 对wav文件提取特征
    wav2fb(dataDir, saveDir, category)

    # usage
    # python process_data.py --data_dir="data/" --save_dir="data/bin" --category="dev"
