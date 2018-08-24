#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : logfbank.py
# @Time     : 2018/8/13 10:43 
# @Software : PyCharm
import librosa
from python_speech_features import logfbank
import numpy as np
import os
from tqdm import tqdm
import re


def get_log_fbank(wavname, winlen=0.025, winstep=0.01, nfilt=40):
    # rate, sig = wav.read(wavname)
    sig, rate = librosa.load(wavname, sr=16000)
    # 归一化 (-1,1)
    try:
        sig = sig.tolist() / max(max(sig), -min(sig))
    except ValueError:
        # 读取文件为空
        return None
    sig = sig.tolist()
    # 将音频裁剪至3s
    if len(sig) > 3 * rate:
        sig = sig[:3 * rate]
    else:
        while len(sig) < 3 * rate:
            sig.append(0)
    sig = np.array(sig)
    try:
        feat = logfbank(sig, rate, winlen=winlen, winstep=winstep, nfilt=nfilt)
    except IndexError:
        return None
    # (N,40)
    return feat


def wav2fb(data_list, save_dir, usage):
    """
    将wav音频文件提取 logfbank 特征后写入.bin二进制文件
    :param data_list: wav文件路径
    :param save_dir: 保存.bin文件路径
    :param usage: 数据集类别
    :return:
    """
    save_path = save_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fileList, label in tqdm(data_list):
        # 创建子文件夹
        sub_dir = os.path.join(save_path, usage, label)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        for wpath in fileList:
            # 计算 fbank 特征
            fbank_feat = get_log_fbank(wpath)
            if fbank_feat is not None:
                file_name = re.search(r"B\S+", wpath).group(0)[:-4]
                fbank_feat.tofile(os.path.join(sub_dir, file_name + ".bin"))


if __name__ == '__main__':
    wavname = "data/train/S0601/BAC009S0601W0324.wav"

    feat = get_log_fbank(wavname)
    print(feat)
    print(feat.shape)
    # train_wavs = "data/train/S0601"
    # file_list = os.listdir(train_wavs)
    # for file in file_list:
    #     feat = get_log_fbank(os.path.join(train_wavs,file))
    #     print(feat.shape)
