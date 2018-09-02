#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : calculate_score.py
# @Time     : 2018/8/17 14:11 
# @Software : PyCharm
from keras import Model
from keras.models import load_model
from absl import flags, app
import os
import re

FLAGS = flags.FLAGS

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

flags.DEFINE_string(
    "checkpoint_path", os.path.join(root_dir, "model/checkpoint-00484-0.99.h5"),
    "the checkpoint model path")
flags.DEFINE_string(
    "model_save_dir", os.path.join(root_dir, "model"),
    "the save model dir")


# 去掉模型Softmax层
def sep_model_weight(checkpoint_path, save_dir):
    model_name = re.search(r"check\S+", checkpoint_path).group(0)[:-3]
    model = load_model(checkpoint_path)
    seq = model.get_layer("sequential_1")
    target_model = Model(inputs=model.input, outputs=seq.get_output_at(1))
    target_model.save(os.path.join(save_dir, model_name + "_notop.h5"))


def main(argv):
    sep_model_weight(FLAGS.checkpoint_path, FLAGS.model_save_dir)


if __name__ == "__main__":
    app.run(main)
