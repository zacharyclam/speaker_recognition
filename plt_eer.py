#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : plt_eer.py
# @Time     : 2018/8/20 10:38 
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# 显示刻度: http://blog.csdn.net/fortware/article/details/51934814

def read_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines


result_infos_path = 'result_infos_eer'
acc_arr = []
recall_arr = []
threshholds = []
results_info_lines = read_file(result_infos_path)
for line in results_info_lines:
    splits = line.strip().split(' ')
    threshhold, acc, recall = splits
    threshholds.append(eval(threshhold))
    acc_arr.append(eval(acc))
    recall_arr.append(eval(recall))

max_x = max(threshholds)
min_x = min(threshholds)
print("minx, maxx", min_x, max_x)
# with legend
fig, ax = plt.subplots()
plt.title("EER curve")

plt.xlabel('threshhold')
plt.ylabel('FA/FR')
yminorLocator = MultipleLocator(0.02)  # 设置y轴的精度
ax.yaxis.set_minor_locator(yminorLocator)  # 设置次刻度线

"""set min and max value for axes"""
ax.set_ylim([0, 0.5])
ax.set_xlim([min_x, max_x])
ax.yaxis.grid(yminorLocator)

ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')
plt.plot(threshholds, acc_arr, '.', label="FA")
plt.plot(threshholds, recall_arr, '.', label="FR")
plt.legend(loc='upper right')

plt.show()  # show the plot on the screen
