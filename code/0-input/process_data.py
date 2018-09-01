#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : process_data.py
# @Time     : 2018/8/11 18:02 
# @Software : PyCharm
import os
from absl import app, flags

try:
    from code.utils.process_wav import wav2fb
    from code.utils.split_data import split_data
except ImportError:
    # ubuntu 下运行会出现 ImportError
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    from utils.process_wav import wav2fb
    from utils.split_data import split_data

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", os.path.join(root_dir, "data/vad_data"), help="the vad data dir")
flags.DEFINE_string("save_dir", os.path.join(root_dir, "data/bin"), help="save fbank vector dir")
flags.DEFINE_string("category", "test", help="the category of data")
flags.DEFINE_float("validata_scale", 0.05, help="the scale of validate data")


def main(argv):
    # 分割数据集
    train_list, validate_list = split_data(FLAGS.data_dir, FLAGS.category, FLAGS.validata_scale)
    # 将特征向量存入 bin 文件中
    # 处理训练数据 分为训练集和验证集
    if FLAGS.validata_scale > 0:
        wav2fb(train_list, os.path.abspath(FLAGS.save_dir), "train")
        wav2fb(validate_list, os.path.abspath(FLAGS.save_dir), "validate")
    # 处理注册和验证时的数据 validata_scale=0
    else:
        wav2fb(train_list, os.path.abspath(FLAGS.save_dir), FLAGS.category)


if __name__ == "__main__":
    app.run(main)
    # usage
    # train on linux
    # nohup python3 -u process_data.py --data_dir="../../data/vad_data" --save_dir="../../data/bin" --category="train" > logs.out 2>&1 &
    # dev
    # python process_data.py --data_dir="../../data/vad_data"  --save_dir="../../data/enrollment_evalution" \
    # --category="dev" --validata_scale=0.0
    # test
    # python process_data.py --data_dir="../../data/vad_data"  --save_dir="../../data/enrollment_evalution" \
    # --category="test" --validata_scale=0.0
