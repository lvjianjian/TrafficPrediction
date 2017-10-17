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
from jampredict.feature.Matrix import matrixsRounding
import numpy as np
from jampredict.model.STResNet import stresnet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from jampredict.feature import Data
from jampredict.utils import Paramater, Metric
from jampredict.utils.Cache import *

np.random.seed(1337)  # for reproducibility

CACHEDATA = True
len_closeness = 5
len_period = 3
len_trend = 1
nb_flow = 1
len_test = 800
use_diff_test = False

nb_residual_unit = 6  # residual unit size
lr = 0.0002  # learning rate
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size

path_result = 'RET'
path_model = 'MODEL'

is_mmn = True  # 是否需要最大最小归一化
hasExternal = True


def build_model(external_dim, x_num, y_num, len_closeness, len_period, len_trend):
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
    # model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


# c_list = [0, 2, 4, 6]
# p_list = [0, 2, 4]
# t_list = [0, 2, 4]
# ex_list = [True, False]
c_list = [5]
p_list = [3]
t_list = [1]
ex_list = [True]


def main():
    all_result = []
    # load data
    for _c in c_list:
        for _p in p_list:
            for _t in t_list:
                for _ex in ex_list:
                    if _c == 0 and _p == 0 and _t == 0:
                        continue
                    len_period = _p
                    len_closeness = _c
                    len_trend = _t
                    hasExternal = _ex

                    print("loading data...")
                    ts = time.time()
                    datapath = os.path.join(Paramater.DATAPATH, "2016", "all")
                    if is_mmn:
                        fname = os.path.join(datapath, 'CACHE',
                                             'TaxiBJ_C{}_P{}_T{}_{}_mmn_speed.h5'.format(len_closeness, len_period,
                                                                                         len_trend,
                                                                                         "External" if hasExternal else "noExternal"))
                    else:
                        fname = os.path.join(datapath, 'CACHE',
                                             'TaxiBJ_C{}_P{}_T{}_{}_speed.h5'.format(len_closeness, len_period,
                                                                                     len_trend,
                                                                                     "External" if hasExternal else "noExternal"))
                    x_num = y_num = 48
                    pkl = fname + '.preprocessing_speed.pkl'
                    fn = "48_48_20_LinearInterpolationFixed"
                    if os.path.exists(fname) and CACHEDATA:
                        X_train, Y_train, X_test, Y_test, mmn, external_dim, \
                        timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = read_cache(fname,
                                                                                                              is_mmn,
                                                                                                              pkl)
                        print("load %s successfully" % fname)
                    else:
                        datapaths = [os.path.join(datapath, fn)]
                        noConditionRegionsPath = os.path.join(datapath, "48_48_20_noSpeedRegion_0.05")
                        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = Data.loadDataFromRaw(
                                paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow,
                                len_closeness=len_closeness,
                                len_period=len_period, len_trend=len_trend
                                , len_test=len_test, maxMinNormalization=is_mmn, preprocess_name=pkl,
                                meta_data=hasExternal,
                                meteorol_data=hasExternal,
                                holiday_data=hasExternal, isComplete=False)
                        if CACHEDATA:
                            cache(fname, X_train, Y_train, X_test, Y_test,
                                  external_dim, timestamp_train, timestamp_test, noConditionRegions, is_mmn, x_num,
                                  y_num,
                                  Paramater.Z_NUM)

                    if use_diff_test:
                        X_test_old = X_test
                        Y_test_old = Y_test
                        import pandas as pd
                        df_diff = pd.read_csv("./data/2016/all/" + fn + "_diff.csv", index_col=0)
                        # 大于200 有335个作为test
                        test_time = df_diff[df_diff["diff"] > 200]["time"].values
                        timestamp_train_dict = dict(zip(timestamp_train, range(len(timestamp_train))))
                        timestamp_test_dict = dict(zip(timestamp_test, range(len(timestamp_test))))
                        new_X_test = []
                        new_Y_test = []
                        if isinstance(X_train, list):
                            for _ in range(len(X_train)):
                                new_X_test.append([])
                        for _test_time in test_time:
                            _test_time = str(_test_time)
                            if (_test_time in timestamp_train_dict):
                                index = timestamp_train_dict[_test_time]
                                if isinstance(X_train, list):
                                    for i in range(len(X_train)):
                                        new_X_test[i].append(X_train[i][index])
                                else:
                                    new_X_test.append(X_train[index])
                                new_Y_test.append(Y_train[index])

                            if (_test_time in timestamp_test_dict):
                                index = timestamp_test_dict[_test_time]
                                if isinstance(X_test_old, list):
                                    for i in range(len(X_test_old)):
                                        new_X_test[i].append(X_test_old[i][index])
                                else:
                                    new_X_test.append(X_test_old[index])
                                new_Y_test.append(Y_test_old[index])

                                # if (_test_time not in timestamp_train_dict and _test_time not in timestamp_test_dict):
                                #     print(_test_time)

                        if isinstance(new_X_test, list):
                            for i in range(len(new_X_test)):
                                new_X_test[i] = np.stack(new_X_test[i], axis=0)
                        else:
                            new_X_test = np.stack(new_X_test, axis=0)
                        new_Y_test = np.stack(new_Y_test, axis=0)

                        # if isinstance(new_X_test, list):
                        #     for i in range(len(new_X_test)):
                        #         print(new_X_test[i].shape)
                        # else:
                        #     print(new_X_test.shape)
                        # print(new_Y_test.shape)
                        X_test = new_X_test
                        Y_test = new_Y_test

                    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::72]])
                    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

                    print('=' * 10)
                    print("compiling model...")
                    print("**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

                    ts = time.time()
                    model = build_model(external_dim, x_num, y_num, len_closeness, len_period, len_trend)
                    hyperparams_name = 'speed.c{}.p{}.t{}.resunit{}.lr{}.{}.{}'.format(
                            len_closeness, len_period, len_trend, nb_residual_unit, lr,
                            "External" if hasExternal else "noExternal",
                            "MMN" if is_mmn else "noMMN")

                    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))

                    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
                    model_checkpoint = ModelCheckpoint(fname_param,
                                                       monitor='val_rmse',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       mode='min',
                                                       save_weights_only=True)

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
                    model.save_weights(os.path.join(path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
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
                    # print("predict", predict)
                    # print("test", Y_test)
                    # print("RMSE:", Metric.RMSE(predict, Y_test, noConditionRegions))
                    # # print("accuracy", Metric.accuracy(predict, Y_test, noConditionRegions))
                    #
                    # print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
                    #
                    # print('=' * 10)
                    # print("training model (cont)...")
                    # ts = time.time()
                    # fname_param = os.path.join(
                    #         path_model, '{}.cont.best.h5'.format(hyperparams_name))
                    # model_checkpoint = ModelCheckpoint(
                    #         fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
                    # history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=2, batch_size=batch_size,
                    #                     callbacks=[
                    #                         model_checkpoint])
                    # pickle.dump((history.history), open(os.path.join(
                    #         path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
                    # model.save_weights(os.path.join(
                    #         path_model, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
                    # print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))
                    #
                    # print('=' * 10)
                    # print('evaluating using the final model')
                    # score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                    #                                                         0] // 48, verbose=0)

                    # if (mmn is not None):
                    #     print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                    #           (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
                    # else:
                    #     print('Train score: %.6f rmse (real): %.6f' %
                    #           (score[0], score[1]))
                    # ts = time.time()
                    # score = model.evaluate(
                    #         X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
                    # if mmn is not None:
                    #     print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                    #           (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
                    # else:
                    #     print('Test score: %.6f rmse (real): %.6f' %
                    #           (score[0], score[1]))
                    #
                    # if not is_mmn:
                    #     predict = model.predict(X_test)
                    # else:
                    #     predict = mmn.inverse_transform(model.predict(X_test))

                    rmse = round(Metric.RMSE(predict, Y_test, noConditionRegions), 5)
                    # np.save("./result/{}_predict_rmse{}".format(hyperparams_name, str(rmse)),
                    #         np.stack([predict, Y_test], axis=0))
                    save_result(predict, Y_test, timestamp_test,
                                "./result/{}_predict_rmse{}".format(hyperparams_name, str(rmse)))

                    print("RMSE:", rmse)

                    # print("accuracy", Metric.accuracy(predict, Y_test, noConditionRegions))
                    all_result.append(
                            "{}c_{}p_{}t_{}External_{}rmse".format(len_closeness, len_period, len_trend, hasExternal,
                                                                   rmse))
                    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))

    for _v in all_result:
        print(_v)


if __name__ == '__main__':
    main()
