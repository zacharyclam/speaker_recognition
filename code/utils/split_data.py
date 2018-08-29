#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : split_data.py
# @Time     : 2018/8/24 20:43 
# @Software : PyCharm
import os


def split_data(data_dir, category, split_scale=0.05):

    data_path = os.path.join(data_dir, category)

    # 获取文件列表
    data_list = []
    for dir in os.listdir(data_path):
        file_list = os.listdir(os.path.join(data_path, dir))
        data_list.append([os.path.join(data_path, dir, file) for file in file_list])

    # (list , label)
    train_list = []
    validate_list = []

    for i, file_list in enumerate(data_list):
        utterence_nums = len(file_list)
        test_nums = utterence_nums * split_scale
        partition = int(utterence_nums - test_nums)
        train_list.append((file_list[:partition], str(i).zfill(4)))
        if split_scale > 0.0:
            validate_list.append((file_list[partition:], str(i).zfill(4)))

    return train_list, validate_list


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    data_dir = os.path.join(root_dir, "data")
    category = "vad_data"
    train_list, validate_list = split_data(data_dir, category, split_scale=0.05)
    for file_list, label in train_list:
        print(len(file_list), label)
