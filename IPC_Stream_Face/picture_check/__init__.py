# -*- coding:utf-8 -*-
"""
检测两张图片的距离
"""
import numpy as np


def check_distance(arr0: np.ndarray, arr1: np.ndarray)->float:
    return np.sqrt(np.sum(np.square(np.subtract(arr0, arr1))))


def check_mean_distance(arr0: np.ndarray, arr1: np.ndarray)->float:
    return check_distance(arr0, arr1)/ np.sum(arr0.shape)


def check_each_part(arr0, arr1, threshold)->tuple:
    assert len(arr0.shape) > 1
    x = arr0.shape[0]//2
    y = arr0.shape[1]//2
    p1 = check_mean_distance(arr0[0:x, 0:y], arr1[0:x, 0:y])
    p2 = check_mean_distance(arr0[x:, 0:y], arr1[x:, 0:y])
    p3 = check_mean_distance(arr0[0:x, y:], arr1[0:x, y:])
    p4 = check_mean_distance(arr0[x:, y:], arr1[x:, y:])
    # print(p1, p2, p3, p4)
    if ((p1 > threshold) & (p4 > threshold)) | ((p2 > threshold) & (p3 > threshold)):
        return 0, 0, arr0.shape[0], arr0.shape[1]
    elif (p1 > threshold) & (p2 > threshold):
        return 0, 0, arr0.shape[0], y
    elif (p2 > threshold) & (p4 > threshold):
        return x, 0, arr0.shape[0], arr0.shape[1]
    elif (p4 > threshold) & (p3 > threshold):
        return 0, y, arr0.shape[0], arr0.shape[1]
    elif (p3 > threshold) & (p1 > threshold):
        return 0, 0, x, arr0.shape[1]
    elif p1 > threshold:
        return 0, 0, x, y
    elif p2 > threshold:
        return x, 0, arr0.shape[0], y
    elif p3 > threshold:
        return 0, y, x, arr0.shape[1]
    elif p4 > threshold:
        return x, y, arr0.shape[0], arr0.shape[1]
    else:
        return -1, -1, -1, -1
