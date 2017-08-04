#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-8-3, 10:57

@Description:

@Update Date: 17-8-3, 10:57
"""

from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape,
    Layer
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import os
import time
import deepst.metrics as metrics
from jampredict.feature.Matrix import matrixsRounding
import numpy as np
from jampredict.model.STResNet import stresnet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from jampredict.feature import Data
from jampredict.utils import Paramater, Metric
from jampredict.utils.Cache import *

CACHEDATA = True
len_closeness = 0
len_period = 3
len_trend = 0
nb_flow = 1
len_test = 800

nb_residual_unit = 6  # residual unit size
lr = 0.0002  # learning rate
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size

path_result = 'RET'
path_model = 'MODEL'

is_mmn = True  # 是否需要最大最小归一化
hasExternal = False


def main():
    # load data
    print("loading data...")
    ts = time.time()
    datapath = os.path.join(Paramater.DATAPATH, "2016", "all")
    if is_mmn:
        fname = os.path.join(datapath, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_{}_mmn_speed.h5'.format(len_closeness, len_period, len_trend,
                                                                         "External" if hasExternal else "noExternal"))
    else:
        fname = os.path.join(datapath, 'CACHE',
                             'TaxiBJ_C{}_P{}_T{}_{}_speed.h5'.format(len_closeness, len_period, len_trend,
                                                                     "External" if hasExternal else "noExternal"))
    x_num = y_num = 48
    pkl = fname + '.preprocessing_speed.pkl'
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, \
        timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = read_cache(fname, is_mmn,
                                                                                              pkl)
        print("load %s successfully" % fname)
    else:
        datapaths = [os.path.join(datapath, "48_48_20_MaxSpeedFillingFixed_5")]
        noConditionRegionsPath = os.path.join(datapath, "48_48_20_noSpeedRegion_0.05")
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = Data.loadDataFromRaw(
            paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow, len_closeness=len_closeness,
            len_period=len_period, len_trend=len_trend
            , len_test=len_test, maxMinNormalization=is_mmn, preprocess_name=pkl,
            meta_data=hasExternal,
            meteorol_data=hasExternal,
            holiday_data=hasExternal, isComplete=False)
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
    print(X_train)

    print "start build model"
    input = Input(shape=(nb_flow * len_period, x_num, y_num))
    conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
    act1 = Activation("relu")(conv1)
    reshape = Reshape((64, 1, x_num, y_num))(act1)

    convLSTM = ConvLSTM2D(nb_filter=1, nb_row=3, nb_col=3, border_mode="same")(reshape)
    # act2 = Activation("relu")(convLSTM)
    # conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(act2)
    main_output = Activation('tanh')(convLSTM)
    model = Model(input=input, output=main_output)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    print "finish build model"

    hyperparams_name = 'testMyModel_speed.c{}.p{}.t{}.resunit{}.lr{}.{}.{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr,
        "External" if hasExternal else "noExternal",
        "MMN" if is_mmn else "noMMN")

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

    if is_mmn:
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Train score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)

    if is_mmn:
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Test score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    if not is_mmn:
        predict = model.predict(X_test)
    else:
        predict = mmn.inverse_transform(model.predict(X_test))
        Y_test = mmn.inverse_transform(Y_test)
    print("predict", predict)
    print("test", Y_test)
    print("RMSE:", Metric.RMSE(predict, Y_test, noConditionRegions))
    # print("accuracy", Metric.accuracy(predict, Y_test, noConditionRegions))

    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    exit(1)


if __name__ == '__main__':
    main()
