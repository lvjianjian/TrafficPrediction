#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-6-20, 22:09

@Description:

@Update Date: 17-6-20, 22:09
"""

from __future__ import print_function

import os
import time

import deepst.metrics as metrics

import numpy as np
from jampredict.model.STResNet import stresnet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from jampredict.feature import Data
from jampredict.utils import Paramater
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

is_mmn = False  # 是否需要最大最小归一化
hasExternal = True


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
                             'TaxiBJ_C{}_P{}_T{}_{}_mmn.h5'.format(len_closeness, len_period, len_trend,
                                                                   "External" if hasExternal else "noExternal"))
    else:
        fname = os.path.join(Paramater.DATAPATH, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_{}.h5'.format(len_closeness, len_period, len_trend,
                                                               "External" if hasExternal else "noExternal"))
    x_num = y_num = 48
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, \
        timestamp_train, timestamp_test, noConditionRegions = read_cache(fname, is_mmn)
        print("load %s successfully" % fname)
    else:
        datapaths = [Paramater.DATAPATH + "48_48_20_LinearInterpolationFixed_condition"]
        noConditionRegionsPath = Paramater.PROJECTPATH + "data/48_48_20_noSpeedRegion_0.05"
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = Data.loadDataFromRaw(
            paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow, len_closeness=len_closeness,
            len_period=len_period, len_trend=len_trend
            , len_test=len_test, maxMinNormalization=is_mmn, preprocess_name='preprocessing.pkl',
            meta_data=hasExternal,
            meteorol_data=hasExternal,
            holiday_data=hasExternal)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test, noConditionRegions, is_mmn, x_num, y_num,
                  Paramater.Z_NUM)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::72]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model(external_dim, x_num=x_num, y_num=y_num)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}.noExternal'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)

    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("\nelapsed time (compiling model): %.3f seconds\n" %
          (time.time() - ts))

    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)

    if mmn is not None:
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Train score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)

    if mmn is not None:
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Test score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join(
        path_model, '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=2, batch_size=batch_size, callbacks=[
        model_checkpoint])
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        path_model, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                            0] // 48, verbose=0)

    if (mmn is not None):
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Train score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    if mmn is not None:
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Test score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))


if __name__ == '__main__':
    main()
