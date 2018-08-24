#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : caculate_score.py
# @Time     : 2018/8/22 17:50 
# @Software : PyCharm
import numpy as np
import os
from absl import flags, app
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())

from utils.csv_util import read_features
from utils.calculate_cds import get_cds


parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "features_dir", default=os.path.join(parent_dir, "results/features"),
    help="the dir of enrolllment")

flags.DEFINE_string(
    "score_dir", default=os.path.join(parent_dir, "results/scores"),
    help="the dir of saving score")


def main(argv):
    validate_dict = read_features(FLAGS.features_dir, "validate")
    strange_dict = read_features(FLAGS.features_dir, "stranger")

    # score n/tp/fp
    with open(os.path.join(FLAGS.score_dir, "score.txt"), "w") as f:
        for val_label, val_feat in tqdm(validate_dict):
            enroll_dict = read_features(FLAGS.features_dir, "enroll")
            distance = [get_cds(val_feat, enroll_feat) for _, enroll_feat in enroll_dict]
            predict_label = np.argmax(distance, axis=0)
            line = str(distance[int(predict_label)]) + " "
            if predict_label == val_label:
                line += "tp\n"
            else:
                line += "fp\n"
            f.write(line)

        for stranger_label, stranger_feat in tqdm(strange_dict):
            enroll_dict = read_features(FLAGS.features_dir, "enroll")
            distance = [get_cds(stranger_feat, enroll_feat) for _, enroll_feat in enroll_dict]
            predict_label = np.argmax(distance, axis=0)
            line = str(distance[int(predict_label)]) + " n\n"
            f.write(line)


if __name__ == "__main__":
    app.run(main)
