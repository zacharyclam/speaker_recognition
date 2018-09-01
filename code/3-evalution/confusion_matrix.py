import numpy as np
import os

from tqdm import tqdm
from absl import flags, app

from code.utils.calculate_cds import get_cds
from code.utils.csv_util import read_features


def confusion_matrix_test(features_dir, matrix_dir):
    validate_dict = read_features(features_dir, "validate")
    # 混淆矩阵
    confusion_matrix = np.zeros((40, 40), dtype=np.int16)
    # 统计
    for val_label, val_feat in tqdm(validate_dict):
        enroll_dict = read_features(features_dir, "enroll")
        distance = [get_cds(val_feat, enroll_feat) for _, enroll_feat in enroll_dict]
        predict_label = np.argmax(distance, axis=0)
        confusion_matrix[val_label][predict_label] += 1
    np.savetxt(os.path.join(matrix_dir, "confusion_matrix.csv"), confusion_matrix, fmt='%d', delimiter=",")


root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "features_dir", os.path.join(root_dir, "results/features"),
    "the enrolled data dir")

flags.DEFINE_string(
    "matrix_dir", default=os.path.join(root_dir, "results"),
    help="the dir of saving confusion matrix")


def main(argv):
    confusion_matrix_test(FLAGS.features_dir, FLAGS.matrix_dir)


if __name__ == "__main__":
    app.run(main)
