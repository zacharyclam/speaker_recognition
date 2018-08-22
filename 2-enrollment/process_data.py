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


def split_data(dataDir, usage, test_scale=0.05):

    dataPath = os.path.join(dataDir, usage)

    # 获取文件列表
    dataList = []
    for dir in os.listdir(dataPath):
        fileList = os.listdir(os.path.join(dataPath, dir))
        dataList.append([os.path.join(dataPath, dir, file) for file in fileList])

    # (list , label)
    train_list = []
    test_list = []

    for i, fileList in enumerate(dataList):
        utterence_nums = len(fileList)
        test_nums = utterence_nums * test_scale
        partition = int(utterence_nums - test_nums)
        train_list.append((fileList[:partition], str(i).zfill(4)))
        test_list.append((fileList[partition:], str(i).zfill(4)))

    return train_list, test_list


def wav2fb(dataDir, saveDir, usage, test_scale=0.05):
    """
    将wav音频文件提取 logfbank 特征后写入.bin二进制文件
    :param dataDir: wav文件路径
    :param saveDir: 保存.bin文件路径
    :param usage: 数据集类别
    :return:
    """
    savePath = saveDir

    train_list, test_list = split_data(dataDir, usage, test_scale)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for fileList, label in tqdm(train_list):
        # 创建子文件夹
        subDir = os.path.join(savePath, "train", label)
        if not os.path.exists(subDir):
            os.makedirs(subDir)
        for wpath in fileList:
            fbank_feat = get_log_fbank(wpath)
            if fbank_feat is not None:
                fileName = re.search(r"B\S+", wpath).group(0)[:-4]
                fbank_feat.tofile(os.path.join(subDir, fileName + ".bin"))

    for fileList, label in tqdm(test_list):
        # 创建子文件夹
        subDir = os.path.join(savePath, "test", label)
        if not os.path.exists(subDir):
            os.makedirs(subDir)
        for wpath in fileList:
            fbank_feat = get_log_fbank(wpath)
            if fbank_feat is not None:
                fileName = re.search(r"B\S+", wpath).group(0)[:-4]
                fbank_feat.tofile(os.path.join(subDir, fileName + ".bin"))


parse = argparse.ArgumentParser()

parse.add_argument("--data_dir", type=str)
parse.add_argument("--save_dir", type=str)
parse.add_argument("--category", type=str, help="the category of data")

if __name__ == "__main__":
    flags, unparsed = parse.parse_known_args(sys.argv[1:])
    # 解析命令行参数
    dataDir = flags.data_dir
    saveDir = flags.save_dir
    category = flags.category

    # 对wav文件提取特征
    wav2fb(dataDir, saveDir, category)

    # usage
    # python3 process_data.py --data_dir="../../untar_data/" --save_dir="data/bin" --category="train"

    # python process_data.py --data_dir="data/" --save_dir="data/bin" --category="dev"
