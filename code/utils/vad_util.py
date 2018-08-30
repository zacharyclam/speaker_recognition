#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : vad_util.py
# @Time     : 2018/8/29 13:37 
# @Software : PyCharm
import numpy as np
from math import log
import librosa


def mse(data):
    return ((data ** 2).mean()) ** 0.5


def dBFS(data):
    mse_data = mse(data)
    if mse_data == 0.0:
        return 0
    max_possible_val = 2 ** 16 / 2
    return 20 * log(mse_data / max_possible_val, 10)


def cut_wav(data, per_f=150):
    num_f = int(len(data) / per_f)
    data = data[:num_f * per_f]
    data = data.reshape((num_f, per_f))
    return data


def remove_silence(source_sound, common_sound, silence_threshold=140, chunk_size=148):
    source_sounds = cut_wav(source_sound, chunk_size)
    common_sounds = cut_wav(common_sound, chunk_size)
    y = []
    for i in range(common_sounds.shape[0]):
        db = -dBFS(common_sounds[i, ...])

        if db < silence_threshold:
            y.append(source_sounds[i])
        # print("db", i, db)
    y = np.array(y)
    y = y.flatten()
    return y


def comman(sound):
    abs_sound = np.abs(sound)
    return sound / np.max(abs_sound)


if __name__ == '__main__':

    wav_data, rate = librosa.load("D:\PythonProject\speakerRecognition\BAC009S0908W0161.wav", sr=16000)

    y = remove_silence(wav_data, wav_data, 139, 300)
    librosa.output.write_wav("c.wav", y, sr=16000)
