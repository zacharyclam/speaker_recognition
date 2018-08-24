#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : process_data.py
# @Time     : 2018/8/11 18:02 
# @Software : PyCharm
from absl import app, flags

from utils.process_wav import wav2fb
from utils.split_data import split_data

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "../../data/", "")
flags.DEFINE_string("save_dir", "../../data/bin", "")
flags.DEFINE_string("category", "test", help="the category of data")
flags.DEFINE_float("validata_scale", 0.05, "the scale os validate data")


def main(argv):
    train_list, validate_list = split_data(FLAGS.data_dir, FLAGS.category, FLAGS.validata_scale)
    wav2fb(train_list, FLAGS.save_dir, "train")
    wav2fb(validate_list, FLAGS.save_dir, "validate")


if __name__ == "__main__":
    app.run(main)

    # usage
    # python3 process_data.py --data_dir="../../untar_data/" --save_dir="data/bin" --category="train"
    # python process_data.py --data_dir="data/" --save_dir="data/bin" --category="dev"
