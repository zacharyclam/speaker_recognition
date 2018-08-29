#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : vad_util.py
# @Time     : 2018/8/29 13:37
# @Software : PyCharm
import os
import re
from tqdm import tqdm
import webrtcvad
from absl import flags, app

try:
    from code.utils.vad_util import read_wave, frame_generator, write_wave, vad_collector
except ImportError:
    # ubuntu 下运行会出现 ImportError
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    from utils.vad_util import read_wave, frame_generator, write_wave, vad_collector


def get_datalist(data_dir, category):
    data_path = os.path.join(data_dir, category)

    # 获取文件列表
    data_list = []
    for idx, dir in enumerate(os.listdir(data_path)):
        file_list = os.listdir(os.path.join(data_path, dir))
        data_list.append(([os.path.join(data_path, dir, file) for file in file_list], str(idx).zfill(4)))
    return data_list


def vad_wav(vad_detector, wav_path, save_dir):
    wav_name = re.search(r"B\S+.$", wav_path).group(0)[:-4]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    audio, sample_rate = read_wave(wav_path)
    frames = frame_generator(30, audio, sample_rate)

    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad_detector, frames)
    for i, segment in enumerate(segments):
        path = os.path.join(save_dir, wav_name + "_{}.wav".format(i))
        write_wave(path, segment, sample_rate)


root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", os.path.join(root_dir, "data"), "")
flags.DEFINE_string("save_dir", os.path.join(root_dir, "data/vad_data"), "")
flags.DEFINE_string("category", "dev", help="the category of data")
flags.DEFINE_integer("mode", 3, "")


def main(args):
    data_list = get_datalist(FLAGS.data_dir, FLAGS.category)
    vad_dector = webrtcvad.Vad(FLAGS.mode)
    save_path = os.path.join(FLAGS.save_dir, FLAGS.category)

    for file_list, label in tqdm(data_list):
        for wav_path in file_list:
            vad_wav(vad_dector, wav_path, os.path.join(save_path, label))


if __name__ == '__main__':
    app.run(main)
