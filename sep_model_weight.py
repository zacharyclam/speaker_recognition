#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : calculate_score.py
# @Time     : 2018/8/17 14:11 
# @Software : PyCharm
from keras import Model
from keras.models import load_model
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_path", "D:\PythonProject\speakerRecognition\model\checkpoint-00763-0.99.h5",
    "the checkpoint model path")
flags.DEFINE_string(
    "model_save_path", "D:\PythonProject\speakerRecognition\model\spk-00152-0.88.h5",
    "the save model path")


def sep_model_weight(checkpoint_path, save_path):
    model = load_model(checkpoint_path)
    seq = model.get_layer("sequential_1")
    target_model = Model(inputs=model.input, outputs=seq.get_output_at(1))
    target_model.save(save_path)


def main(argv):
    sep_model_weight(FLAGS.checkpoint_path, FLAGS.model_save_path)


if __name__ == "__main__":
    app.run(main)
