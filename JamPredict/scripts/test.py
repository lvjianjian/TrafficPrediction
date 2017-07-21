#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-19, 19:10

@Description:

@Update Date: 17-7-19, 19:10
"""

from __future__ import print_function

from sys import path

print(path)
print(__name__)
print(__package__)
import cPickle as pickle
import os
import time

import deepst.metrics as metrics
import h5py
import numpy as np
from jampredict.model.STResNet import stresnet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from jampredict.feature.Matrix import matrixsRounding
from jampredict.feature import Data
from jampredict.utils import Paramater
from jampredict.utils import Metric
from jampredict.utils.Cache import *

np.random.seed(1337)  # for reproducibility

CACHEDATA = True
len_closeness = 3
len_period = 1
len_trend = 1
nb_flow = 1
len_test = 300

nb_residual_unit = 6  # residual unit size
lr = 0.0002  # learning rate
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size

path_result = 'RET'
path_model = 'MODEL'
is_mmn = False


def build_model(external_dim, x_num, y_num):
    c_conf = (len_closeness, nb_flow, y_num,
              x_num) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, y_num,
              x_num) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, y_num,
              x_num) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit, isRegression=is_mmn)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def main():
    # load data
    print("loading data...")
    ts = time.time()
    if is_mmn:
        fname = os.path.join(Paramater.DATAPATH, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_noExternal_mmn.h5'.format(len_closeness, len_period, len_trend))
    else:
        fname = os.path.join(Paramater.DATAPATH, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_noExternal.h5'.format(len_closeness, len_period, len_trend))

    x_num = y_num = 48
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions = read_cache(
            fname, is_mmn)
        print("load %s successfully" % fname)
    else:
        datapaths = [Paramater.DATAPATH + "48_48_20_LinearInterpolationFixed_condition"]
        noConditionRegionsPath = Paramater.PROJECTPATH + "data/48_48_20_noSpeedRegion_0.05"
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions = Data.loadDataFromRaw(
            paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow, len_closeness=len_closeness,
            len_period=len_period, len_trend=len_trend
            , len_test=len_test, preprocess_name='preprocessing.pkl', meta_data=False, meteorol_data=False,
            holiday_data=False)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test, noConditionRegions, is_mmn)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::72]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model(external_dim, x_num=x_num, y_num=y_num)

    model.load_weights(
        Paramater.PROJECTPATH + "/MODEL/c3.p1.t1.resunit6.lr0.0002.noExternal.cont.best.h5")
    predict = matrixsRounding(model.predict(X_test))
    print("RMSE:", Metric.RMSE(predict, Y_test, noConditionRegions))
    print("accuracy", Metric.accuracy(predict, Y_test, noConditionRegions))

    # score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    # print(score)


if __name__ == '__main__':
    main()
