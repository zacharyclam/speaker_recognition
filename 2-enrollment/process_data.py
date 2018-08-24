#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : process_data.py
# @Time     : 2018/8/11 18:02 
# @Software : PyCharm
import os
import re
from tqdm import tqdm
from get_log_fbank import get_log_fbank
import argparse
import sys


def split_data(dataDir, usage):

    dataPath = os.path.join(dataDir, usage)

    # 获取文件列表
    dataList = []
    for idx, dir in enumerate(os.listdir(dataPath)):
        fileList = os.listdir(os.path.join(dataPath, dir))
        dataList.append(([os.path.join(dataPath, dir, file) for file in fileList], idx))

    return dataList


def wav2fb(dataDir, saveDir, usage):
    """
    将wav音频文件提取 logfbank 特征后写入.bin二进制文件
    :param dataDir: wav文件路径
    :param saveDir: 保存.bin文件路径
    :param usage: 数据集类别
    :return:
    """
    savePath = saveDir

    data_list = split_data(dataDir, usage)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for fileList, label in tqdm(data_list):
        # 创建子文件夹
        subDir = os.path.join(savePath, category, str(label).zfill(4))
        if not os.path.exists(subDir):
            os.makedirs(subDir)
        for wpath in fileList:
            fbank_feat = get_log_fbank(wpath)
            if fbank_feat is not None:
                fileName = re.search(r"B\S+", wpath).group(0)[:-4]
                fbank_feat.tofile(os.path.join(subDir, fileName + ".bin"))


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
