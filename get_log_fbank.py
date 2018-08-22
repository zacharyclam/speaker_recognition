#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : logfbank.py
# @Time     : 2018/8/13 10:43 
# @Software : PyCharm
import librosa
from python_speech_features import logfbank
import numpy as np


def get_log_fbank(wavname, winlen=0.025, winstep=0.01, nfilt=40):
    # rate, sig = wav.read(wavname)
    sig, rate = librosa.load(wavname, sr=16000)
    # 归一化 (-1,1)
    try:
        sig = sig.tolist() / max(max(sig), -min(sig))
    except ValueError:
        # 读取文件为空
        print(wavname)
        return None
    sig = sig.tolist()
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
