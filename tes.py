#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tes.py
# @Time     : 2018/8/20 11:43 
# @Software : PyCharm

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import re

'''
calculate each rate
'''


def cal_rate(result, num, thres):
    all_number = len(result[0])
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        disease = result[0][item, num]
        if disease >= thres:
            disease = 1
        if disease == 1:
            if result[1][item, num] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[1][item, num] == 0:
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
    # print accracy, precision, TPR, TNR, FNR, FPR
    return accracy, precision, TPR, TNR, FNR, FPR


disease_class = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                 'Pneumothorax']
style = ['r-', 'g-', 'b-', 'y-', 'r--', 'g--', 'b--', 'y--']
'''
plot roc and calculate AUC/ERR, result: (prob, label) 
'''
prob = np.random.rand(100, 8)
label = np.where(prob >= 0.5, prob, 0)
label = np.where(label < 0.5, label, 1)
count = np.count_nonzero(label)
label = np.zeros((100, 8))
label[1:20, :] = 1
print(label)
print(prob)
print(count)

for clss in range(len(disease_class)):
    threshold_vaule = sorted(prob[:, clss])
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
    # calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob, label), clss, threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR
    # print TPR_array
    # print FPR_array
    AUC = np.trapz(TPR_array, FPR_array)
    threshold = np.argmin(abs(FNR_array - FPR_array))
    EER = (FNR_array[threshold] + FPR_array[threshold]) / 2
    print('disease %10s threshold : %f' % (disease_class[clss], threshold))
    print('disease %10s accracy : %f' % (disease_class[clss], accracy_array[threshold]))
    print('disease %10s EER : %f AUC : %f' % (disease_class[clss], EER, -AUC))
    plt.plot(FPR_array, TPR_array, style[clss], label=disease_class[clss])
plt.title('roc')
plt.xlabel('FPR_array')
plt.ylabel('TPR_array')
plt.legend()
plt.show()
