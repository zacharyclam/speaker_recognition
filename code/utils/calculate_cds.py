#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : calculate_cds.py
# @Time     : 2018/8/24 14:32 
# @Software : PyCharm
import numpy as np


def get_cds(a, b):
    """
    返回归一化后的余弦距离，得分CDS越接近1越好
    :param a: shape[n,-1]
    :param b: shape[n, -1]
    :return:
    """

    num = float(a.dot(b.T))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cds = num / denom  # 余弦值
    cds = 0.5 + 0.5 * cds  # 归一化
    return cds
