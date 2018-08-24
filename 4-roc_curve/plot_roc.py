#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : plot_roc.py
# @Time     : 2018/8/22 20:58 
# @Software : PyCharm
import os
import numpy as np
import matplotlib.pylab as plt
from absl import app, flags

FLAGS = flags.FLAGS

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

flags.DEFINE_string(
    "save_plot_dir", os.path.join(parent_dir, "results/plots"),
    "the generate plots image dir")

flags.DEFINE_string(
    "plot_name", "plt_roc_spk-00298-0.99",
    "the roc image's name")

flags.DEFINE_string(
    "score_dir", os.path.join(parent_dir, "results/scores"),
    "the score txt dir")


def cal_rate(score_dict, thres):
    all_number = len(score_dict)
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for score, label in score_dict:
        disease = score
        if disease >= thres:
            disease = 1
        if disease == 1:
            if label == "tp":
                TP += 1
            else:
                FP += 1
        else:
            if label == "n":
                TN += 1
            else:
                FN += 1
    # print TP+FP+TN+FN
    accracy = float(TP + FP) / float(all_number)
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    TPR = float(TP) / float(TP + FN)
    TNR = float(TN) / float(FP + TN)
    FNR = float(FN) / float(TP + FN)
    FPR = float(FP) / float(FP + TN)

    return accracy, precision, TPR, TNR, FNR, FPR


def plot_roc(score_list, save_dir, plot_name):

    save_path = os.path.join(save_dir, plot_name + ".jpg")
    # 按照 score 排序
    threshold_value = sorted([score for score, _ in score_list])

    threshold_num = len(threshold_value)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)

    # calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR = cal_rate(score_list, threshold_value[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR

    AUC = np.trapz(TPR_array, FPR_array)
    threshold = np.argmin(abs(FNR_array - FPR_array))
    EER = (FNR_array[threshold] + FPR_array[threshold]) / 2
    # print('EER : %f AUC : %f' % (EER, -AUC))
    plt.plot(FPR_array, TPR_array)

    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.text(0.4, 0, s="EER :{} AUC :{}".format(round(EER, 2), round(-AUC, 2)), fontsize=10)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def main(argv):
    score_list = []

    with open(os.path.join(FLAGS.score_dir, "score.txt"), "r") as f:
        for line in f:
            score, label = line.split(" ")
            score_list.append([float(score), label.rstrip("\n")])

    plot_roc(score_list, FLAGS.save_plot_dir, FLAGS.plot_name)

    # os.path.join(parent_dir, "results/plots", "plt_roc_spk-00344-0.98.jpg")


if __name__ == "__main__":
    app.run(main)
