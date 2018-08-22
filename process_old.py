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
    """

    :param dataDir: 文件存储路径
    :param usage: 数据集类别
    :param enroll_num: 注册使用语句条数
    :param val_num: 验证使用语句条数
    :return:
    """
    dataPath = os.path.join(dataDir, usage)

    # 获取文件列表
    dataList = []
    for dir in os.listdir(dataPath):
        fileList = os.listdir(os.path.join(dataPath, dir))
        dataList.append([os.path.join(dataPath, dir, file) for file in fileList])

    # (list , label)
    train_data = []
    test_data = []

    for i, fileList in enumerate(dataList):
        utterence_nums = len(fileList)
        test_nums =utterence_nums * test_scale
        partition = utterence_nums - test_nums
        train_data.append((fileList[:partition], i))
        test_data.append((fileList[partition:], i))

    return train_data,test_data

def wav2fb(dataDir, saveDir, usage, test_scale=0.05):
    """
    将wav音频文件提取 logfbank 特征后写入.bin二进制文件
    :param dataDir: wav文件路径
    :param saveDir: 保存.bin文件路径
    :param usage: 数据集类别
    :return:
    """
    dataPath = os.path.join(dataDir, usage)
    savePath = os.path.join(saveDir, usage)

    train_data, test_data = split_data(dataDir,usage,test_scale)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # 获取文件列表
    dataDict = {}
    for dir in os.listdir(dataPath):
        fileList = os.listdir(os.path.join(dataPath, dir))
        dataDict[dir] = [os.path.join(dataPath, dir, file) for file in fileList]

    for label, fileList in tqdm(dataDict.items()):
        # 创建子文件夹
        subDir = os.path.join(savePath, label)
        if not os.path.isdir(subDir):
            os.mkdir(subDir)
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

    # python process_data.py --data_dir="data/" --save_dir="data/bin" --category="test"
