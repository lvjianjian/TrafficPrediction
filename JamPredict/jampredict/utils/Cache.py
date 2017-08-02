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


def read_cache(fname, is_mmn,preprocess_fname='preprocessing.pkl'):
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


if __name__ == '__main__':
    # h5 = h5py.File("test.h5", 'w')
    # h5.create_dataset('x_num', data=1)
    # h5.close()

    f = h5py.File("test.h5", 'r')
    x_num = f['x_num'].value
    print x_num, type(x_num)
    print int(x_num)
    f.close()
