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
    sig, rate = librosa.load(wavname, sr=16000)
    # 归一化 (-1,1)
    try:
        sig = sig.tolist() / max(max(sig), -min(sig))
    except ValueError:
        # 读取文件为空
        return None
    sig = np.array(sig)
    section_nums = len(sig) // rate
    # 将音频切分为1秒1段
    audio_list = [sig[partition * rate:(partition + 1)*rate]for partition in range(section_nums)]

    try:
        feat = [logfbank(audio, rate, winlen=winlen, winstep=winstep, nfilt=nfilt) for audio in audio_list]
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
            fbank_feats = get_log_fbank(wpath)
            if fbank_feats is not None:
                file_name = re.search(r"B\S+", wpath).group(0)[:-4]
                for idx, fbank in enumerate(fbank_feats):
                    fbank.tofile(os.path.join(sub_dir, file_name + "_{}.bin".format(idx)))


if __name__ == '__main__':
    wavname = "D:\\PythonProject\\speakerRecognition\\data\\train\\S0002\\BAC009S0002W0122.wav"

    feat = get_log_fbank(wavname)
    print(feat)
    for f in feat:
        print(f.shape)


    # from split_data import split_data
    # root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    #
    # data_dir = os.path.join(root_dir, "data")
    # category = "vad_data"
    # save_dir = os.path.join(root_dir, "data/vad_bin")
    # train_list, validate_list = split_data(data_dir, category, split_scale=0.05)
    # wav2fb(train_list, save_dir, "train")
