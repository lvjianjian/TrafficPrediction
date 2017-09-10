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
month = "all"
path_result = 'RET'
path_model = 'MODEL'

is_mmn = True  # 是否需要最大最小归一化
hasExternal = False

from keras import backend as K
from keras.engine.topology import Layer


class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             border_mode="same")(activation)

    return f


def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _residual_unit(nb_filter, subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)

    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  subsample=init_subsample)(input)
        return input

    return f


def main():
    # load data
    print("loading data...")
    ts = time.time()

    datapath = os.path.join(Paramater.DATAPATH, "2016", month)
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
    main_inputs = []
    main_outputs = []

    if len_closeness > 0:
        input = Input(shape=(nb_flow * len_closeness, x_num, y_num))
        main_inputs.append(input)
        # Conv1
        conv1 = Convolution2D(
                nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
        # [nb_residual_unit] Residual Units
        residual_output = ResUnits(_residual_unit, nb_filter=64,
                                   repetations=nb_residual_unit)(conv1)
        # Conv2
        activation = Activation('relu')(residual_output)
        conv2 = Convolution2D(
                nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
        main_outputs.append(conv2)

        # input = Input(shape=(nb_flow * len_closeness, x_num, y_num))
        # main_inputs.append(input)
        # # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
        # # act1 = Activation("relu")(conv1)
        # reshape = Reshape((len_closeness, nb_flow, x_num, y_num))(input)
        # convLSTM = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode="same")(reshape)
        # act2 = Activation("relu")(convLSTM)
        # conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(act2)
        # main_outputs.append(conv2)

    if len_period > 0:
        input = Input(shape=(nb_flow * len_period, x_num, y_num))
        main_inputs.append(input)
        # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
        # act1 = Activation("relu")(conv1)
        reshape = Reshape((len_period, nb_flow, x_num, y_num))(input)
        convLSTM = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode="same")(reshape)
        act2 = Activation("relu")(convLSTM)
        conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(act2)
        main_outputs.append(conv2)

    if len_trend > 0:
        input = Input(shape=(nb_flow * len_trend, x_num, y_num))
        main_inputs.append(input)
        # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
        # act1 = Activation("relu")(conv1)
        reshape = Reshape((len_trend, nb_flow, x_num, y_num))(input)
        convLSTM = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode="same")(reshape)
        act2 = Activation("relu")(convLSTM)
        conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(act2)
        main_outputs.append(conv2)

    if len(main_outputs) == 1:
        main_output = main_outputs[0]
    else:
        new_outputs = []
        for output in main_outputs:
            new_outputs.append(iLayer()(output))
        main_output = merge(new_outputs, mode='sum')

    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * x_num * y_num)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, x_num, y_num))(activation)
        main_output = merge([main_output, external_output], mode='sum')

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    print "finish build model"

    hyperparams_name = 'testMyModel2_speed.c{}.p{}.t{}.resunit{}.lr{}.{}.{}'.format(
            len_closeness, len_period, len_trend, nb_residual_unit, lr,
            "External" if hasExternal else "noExternal",
            "MMN" if is_mmn else "noMMN")

    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=4, mode='min')
    model_checkpoint = ModelCheckpoint(
            fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("\nelapsed time (compiling model): %.3f seconds\n" %
          (time.time() - ts))

    print('=' * 10)
    print "X_train shape:", X_train.shape
    print "X_test shape:", X_test.shape

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
