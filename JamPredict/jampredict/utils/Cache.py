#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-20, 13:49

@Description:

@Update Date: 17-7-20, 13:49
"""

import h5py
import cPickle as pickle
import numpy as np


def read_cache(fname, is_mmn, preprocess_fname='preprocessing.pkl'):
    if (is_mmn):
        mmn = pickle.load(open(preprocess_fname, 'rb'))
    else:
        mmn = None
    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    if num == 1:
        X_train = X_train[0]
        X_test = X_test[0]
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    noConditionRegions = f['noConditionRegions'].value
    x_num = f['x_num'].value
    y_num = f['y_num'].value
    z_num = f['z_num'].value
    f.close()

    noConditionRegionsSet = set()
    for i in range(noConditionRegions.shape[0]):
        noConditionRegionsSet.add((noConditionRegions[i][0], noConditionRegions[i][1]))
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegionsSet, int(
            x_num), int(y_num), int(z_num)


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test, noConditionRegions,
          is_mmn, x_num, y_num, z_num):
    # if is_mmn:
    #     fname = fname.replace(".h5", "_mmn.h5")
    h5 = h5py.File(fname, 'w')
    if (type(X_train) is np.ndarray):
        X_train = [X_train]
        X_test = [X_test]
    h5.create_dataset('num', data=len(X_train))
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    if type(noConditionRegions) == set:
        noConditionRegions = list(noConditionRegions)
    h5.create_dataset('noConditionRegions', data=noConditionRegions)
    h5.create_dataset('x_num', data=x_num)
    h5.create_dataset('y_num', data=y_num)
    h5.create_dataset('z_num', data=z_num)
    h5.close()


def save_result(predict, real, time_stamp, fname):
    if ".h5" not in fname:
        fname += ".h5"

    h5 = h5py.File(fname, 'w')
    h5.create_dataset('predict', data=predict)
    h5.create_dataset('real', data=real)
    h5.create_dataset('time', data=time_stamp)
    h5.close()


def read_result(fname):
    if ".h5" not in fname:
        fname += ".h5"

    h5 = h5py.File(fname, 'r')
    predict = h5['predict'].value
    real = h5['real'].value
    time = h5['time'].value
    h5.close()
    return predict, real, time


if __name__ == '__main__':
    # h5 = h5py.File("test.h5", 'w')
    # h5.create_dataset('x_num', data=1)
    # h5.close()

    # f = h5py.File("test.h5", 'r')
    # x_num = f['x_num'].value
    # print x_num, type(x_num)
    # print int(x_num)
    # f.close()
    import pandas as pd

    df_diff = pd.read_csv("../../data/2016/all/48_48_20_MaxSpeedFillingFixed_5_diff.csv", index_col=0)
    top_diffs = df_diff.time[:3000].values
    print top_diffs
    exit(1)
    p1, r1, t1 = read_result("../../result/speed.c5.p3.t1.resunit6.lr0.0002.External.MMN_predict_rmse2.24067")
    import numpy as np

    p2, r2, t2 = read_result("../../result/testMyModel4(ernn_h64_l2_step5)_speed.c5.p3.t1.resunit6.lr0.0002.External.MMN_predict_rmse2.22299")
    from Metric import RMSE
    from jampredict.feature.Data import get_no_speed_region

    nospeed = get_no_speed_region("../../data/2016/all/48_48_20_noSpeedRegion_0.05", 48)
    index1 = np.in1d(t1, top_diffs)
    index2 = np.in1d(t2, top_diffs)

    print "resnet", RMSE(p1[index1], r1[index1], nospeed), "compute ", index1.sum()
    print "testmodel4", RMSE(p2[index2], r2[index2], nospeed), "compute ", index2.sum()
