#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : calcuate_eer.py
# @Time     : 2018/8/21 20:37 
# @Software : PyCharm

import os


import numpy as np
from keras.models import load_model
import re
from tqdm import tqdm
import python_speech_features as psf

from get_log_fbank import get_log_fbank


def split_data(data_dir, usage, enroll_sentence_nums=20, val_sentence_nums=100):
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
    enroll_tuple = []
    val_tuple = []

    for i, file_list in enumerate(data_list):
        enroll_tuple.append((file_list[:enroll_sentence_nums], i))
        val_tuple.append((file_list[enroll_sentence_nums:enroll_sentence_nums + val_sentence_nums], i))

    return enroll_tuple, val_tuple


def get_strangerlist(data_dir, sentence_nums=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 获取文件列表
    data_list = []
    for dir in os.listdir(data_dir):
        file_list = os.listdir(os.path.join(data_dir, dir))
        data_list.append([os.path.join(data_dir, dir, file) for file in file_list])

    stranger_list = []
    for i, file_list in enumerate(data_list):
        stranger_list.append((file_list[:sentence_nums], str(i).zfill(4)))

    return stranger_list


def wav2fb(dataTuple, saveDir, category):
    # enrol
    # - - -
    # val
    # - - -
    savePath = os.path.join(saveDir, category)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for (file_list, label) in tqdm(dataTuple):
        for file in file_list:
            subDir = os.path.join(savePath, str(label).zfill(4))
            if not os.path.isdir(subDir):
                os.mkdir(subDir)

            fbank_feat = get_log_fbank(file)
            # fbank_feat = getWavFeat(file)
            if fbank_feat is not None:
                fileName = re.search(r"B\S+", file).group(0)[:-4]
                # print(os.path.join(subDir, fileName + ".bin"))
                fbank_feat.tofile(os.path.join(subDir, fileName + ".bin"))


# enroll_data,val_data = split_data(dataDir, usage)
# wav2fb(enroll_data,save_dir,"enroll")
# wav2fb(val_data,save_dir,"val")

def getList(save_dir, category):
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
        for subname in sub_dir:
            subpath = os.path.join(data_dir, subname)
            for filename in os.listdir(subpath):
                # 文件路径 标签
                line = os.path.join(subpath, filename) + " " + subname + "\n"
                f.write(line)


def getCDS(a, b):
    """
    返回归一化后的余弦距离，得分CDS越接近1越好
    :param a: shape[1,-1]
    :param b: shape[1, -1]
    :return:
    """

    num = a.dot(b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cds = num / denom  # 余弦值
    cds = 0.5 + 0.5 * cds  # 归一化
    return cds


def getFeatures(model, dataListDir, category, mean=True, sentence_nums=20):
    def caculateFeatures(model, fb_input, mean=True):
        features = model.predict(fb_input)
        features = np.array(features)
        if mean:
            # (1,256)
            return np.mean(features, axis=0)
        else:
            # (N,256)
            return features

    data_path = os.path.join(dataListDir, category + "_list.txt")
    people_dict = {}
    with open(data_path) as f:
        X = []
        cnt = 0
        for line in f:
            bin_path, label = line.split(" ")
            x = np.fromfile(bin_path, dtype=np.float)
            X.append(x)

            cnt += 1
            if cnt % sentence_nums == 0:
                features = caculateFeatures(model, np.array(X)[:, :, np.newaxis], mean)
                cnt = 0
                X = []
                people_dict[label] = features

    return people_dict


def enrollment():
    # 获取上级目录
    parent_dir = os.path.dirname(os.path.abspath("__file__"))
    data_dir = os.path.join(parent_dir, "data")

    usage = "dev"
    save_dir = "eer_data"
    stranger_dir = os.path.join(save_dir, "stranger")
    enrolled_dir = os.path.join(save_dir, "enrolled")

    model = load_model(weight_path)

    # 分割 注册人 数据集
    enroll_data, val_data = split_data(data_dir, usage, enroll_sentence_nums=20, val_sentence_nums=3)

    wav2fb(enroll_data, enrolled_dir, "enroll")
    wav2fb(val_data, enrolled_dir, "val")
    getList(enrolled_dir, "enroll")
    getList(enrolled_dir, "val")


if __name__ == "__main__":
    weight_path = "spk5.h5"
    # 获取上级目录
    parent_dir = os.path.dirname(os.path.abspath("__file__"))

    data_dir = os.path.join(parent_dir, "data")

    usage = "dev"
    save_dir = "eer_data"
    stranger_dir = os.path.join(save_dir, "stranger")
    enrolled_dir = os.path.join(save_dir, "enrolled")

    model = load_model(weight_path)

    # 分割 注册人 数据集
    enroll_data, val_data = split_data(data_dir, usage, enroll_sentence_nums=20, val_sentence_nums=3)

    wav2fb(enroll_data, enrolled_dir, "enroll")
    wav2fb(val_data, enrolled_dir, "val")
    getList(enrolled_dir, "enroll")
    getList(enrolled_dir, "val")

    # 生成陌生人数据集
    # stranger_list = get_strangerlist("data/test", sentence_nums=5)
    # wav2fb(stranger_list, save_dir, "stranger")
    # getList(save_dir, "stranger")
    #
    # score_path = "score.txt"
    #
    # # 测试
    # enroll_people = getFeatures(model, enrolled_dir, category="enroll", mean=True, sentence_nums=20)
    # val_people = getFeatures(model, enrolled_dir, category="val", mean=True, sentence_nums=2)
    # stranger_people = getFeatures(model, save_dir, category="stranger", mean=True, sentence_nums=5)
    #
    # f = open(score_path, "w")
    # for (enrol_label, enroll_feature), (val_label, val_feature) in zip(enroll_people.items(), val_people.items()):
    #     distance = getCDS(enroll_feature, val_feature)
    #     line = str(round(distance, 3)) + " target\n"
    #     f.write(line)
    #
    # for (_, stranger_feature) in stranger_people.items():
    #     distance = max([getCDS(stranger_feature, enroll_feature) for (_, enroll_feature) in enroll_people.items()])
    #     line = str(round(distance, 3)) + " nontarget\n"
    #     f.write(line)
    #
    # f.close()