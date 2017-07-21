#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-19, 19:54

@Description:

@Update Date: 17-7-19, 19:54
"""

import numpy as np
import math


def RMSE(predict, real, noConditionRegions):
    '''

    :param predict: ndarray (samplesize,z_num,x_num,y_num)
    :param real: 同上
    :param noConditionRegions: list[tuple1,tuple2,..] tuple = (x,y)
    :return:
    '''
    assert predict.shape == real.shape
    sample_size = predict.shape[0]
    z_num = predict.shape[1]
    x_num = predict.shape[2]
    y_num = predict.shape[3]
    rmse = 0
    count = 0
    for i in range(sample_size):
        for j in range(z_num):
            for x in range(x_num):
                for y in range(y_num):
                    if (x, y) in noConditionRegions:
                        continue
                    rmse += math.pow(predict[i, j, x, y] - real[i, j, x, y], 2)
                    count += 1
    return math.sqrt(rmse / count)


def accuracy(predict, real, noConditionRegions):
    '''
    精度
    :param predict:
    :param real:
    :param noConditionRegions:
    :return:
    '''
    assert predict.shape == real.shape
    sample_size = predict.shape[0]
    z_num = predict.shape[1]
    x_num = predict.shape[2]
    y_num = predict.shape[3]
    accuracy = 0
    count = 0
    for i in range(sample_size):
        for j in range(z_num):
            for x in range(x_num):
                for y in range(y_num):
                    if (x, y) in noConditionRegions:
                        continue
                    accuracy += (1 if (predict[i, j, x, y] == real[i, j, x, y]) else 0)
                    count += 1
    return float(accuracy) / count


if __name__ == '__main__':
    a = np.ndarray(shape=(300, 1, 10, 10))
    a[0, 0, 0, 0] = 1
    a[0, 0, 0, 1] = 1
    b = np.ndarray(shape=(300, 1, 10, 10))
    list = [(0, 0), (1, 1)]
    print accuracy(a, b, list)
