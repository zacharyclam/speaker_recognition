#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : process_data.py
# @Time     : 2018/8/11 18:02 
# @Software : PyCharm
from absl import flags, app

from utils.process_wav import wav2fb
from utils.split_data import split_data

FLAGS = flags.FLAGS


flags.DEFINE_string("data_dir", "D:\PythonProject\speakerRecognition\data", "")
flags.DEFINE_string("save_dir", "D:\PythonProject\speakerRecognition\data\enrollment_evalution", "")
flags.DEFINE_string("category", "test", "the category of data")


def main(argv):
    # 将注册人的wav文件转成特征向量
    enrolled_list, _ = split_data(FLAGS.data_dir, FLAGS.category, 0.0)
    # dev test
    wav2fb(enrolled_list, FLAGS.save_dir, FLAGS.category)


if __name__ == "__main__":
    app.run(main)
