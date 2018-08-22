#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : unzip_tar.py
# @Time     : 2018/8/17 9:01 
# @Software : PyCharm
import tarfile
import os
import tqdm


# 解压tar.gz文件到文件夹
def untar_dir(srcname, dstPath):
    tarHandle = tarfile.open(srcname, "r:gz")
    tarHandle.extractall(dstPath)
    tarHandle.close()


def untar(dataDir, saveDir):
    """
    
    @param dataDir:
    @param saveDir:
    @return:
    """
    # 将音频文件解压
    for root, dir, file in os.walk(dataDir):
        for filename in tqdm(file):
            untar_dir(os.path.join(root, filename), saveDir)


if __name__ == "__main__":
    dataDir = '../../data/data_aishell/wav/'
    saveDir = '../../untar_data/'
    untar(dataDir, saveDir)
