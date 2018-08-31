#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : vad_util.py
# @Time     : 2018/8/29 13:37
# @Software : PyCharm
import os
import re
from tqdm import tqdm
from absl import flags, app
import librosa
import numpy as np

try:
    from code.utils.vad_util import remove_silence
except ImportError:
    # ubuntu 下运行会出现 ImportError
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    from utils.vad_util import remove_silence


def get_datalist(data_dir, category):
    data_path = os.path.join(data_dir, category)

    # 获取文件列表
    data_list = []
    for idx, dir in enumerate(os.listdir(data_path)):
        file_list = os.listdir(os.path.join(data_path, dir))
        data_list.append(([os.path.join(data_path, dir, file) for file in file_list], str(idx).zfill(4)))
    return data_list


def vad_wav(wav_path, save_dir, sr=16000):
    wav_name = re.search(r"B\S+.$", wav_path).group(0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wav_data, rate = librosa.load(wav_path, sr)

    y = remove_silence(wav_data, wav_data, 139, 300)
    # 写入文件
    librosa.output.write_wav(os.path.join(save_dir, wav_name), y, rate)


root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", os.path.join(root_dir, "data"), "")
flags.DEFINE_string("save_dir", os.path.join(root_dir, "data/vad_data"), "")
flags.DEFINE_string("category", "dev", help="the category of data")


def main(args):
    data_list = get_datalist(FLAGS.data_dir, FLAGS.category)
    save_path = os.path.join(FLAGS.save_dir, FLAGS.category)
    #
    # for file_list, label in tqdm(data_list):
    #     for wav_path in file_list:
    #         vad_wav(wav_path, os.path.join(save_path, label))
    wav_data, rate = librosa.load("recorded_audio.wav", sr=16000)

    y = remove_silence(wav_data, wav_data, 139, 300)
    # 写入文件
    librosa.output.write_wav("change.wav", y, rate)


if __name__ == '__main__':
    app.run(main)
    # nohup python3 -u vad.py --save_dir="../../data/vad_data"  --data_dir="../../../../untar_data" --category="train" > logs.out 2>&1 &

