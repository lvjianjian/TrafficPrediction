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
    Layer,
    Embedding,
    merge,
    Conv2D,
    Lambda
)
from keras.utils import np_utils
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
from keras import initializers, activations
from keras.engine.topology import InputSpec
import functools
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nnet import conv2d

import theano
import theano.tensor as T

CACHEDATA = True
len_closeness = 5
len_period = 3
len_trend = 1
nb_flow = 1
len_test = 800
use_diff_test = True

nb_residual_unit = 6  # residual unit size
lr = 0.0002  # learning rate
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size

month = "all"
path_result = 'RET'
path_model = 'MODEL'

step = 10
error_hidden_dim = 32
l = 2

is_mmn = True  # 是否需要最大最小归一化
hasExternal = True


class eRNN(Layer):
    def __init__(self, hidden_dim, output_dim_shape, l, return_sequences=False, **kwargs):
        self.n_past_error = l
        self.output_dim = output_dim_shape
        if isinstance(self.output_dim, int):
            self.output_dim = tuple([self.output_dim])
        self.error_hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        super(eRNN, self).__init__(**kwargs)

    def build(self, input_shape):  # [(none,step,x_f),(none,step,y_f)]
        print "build", input_shape
        assert input_shape[0][1] == input_shape[1][1]
        self.input_spec = [InputSpec(shape=sh) for sh in input_shape]
        # self.input_x_dim = input_shape[0][2]
        # self.input_y_dim = input_shape[1][2]
        # self.input_length = input_shape[1][1]
        self.states = [None]
        self.W_shape = (3, 3, self.n_past_error, self.error_hidden_dim)
        self.W = self.add_weight(self.W_shape,
                                 initializer=initializers.get('glorot_uniform'),
                                 name='{}_W'.format(self.name),
                                 regularizer=None,
                                 constraint=None)
        self.b = self.add_weight((self.error_hidden_dim,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=None)

        self.output_W_shape = (1, 1, self.error_hidden_dim, self.output_dim[0])

        self.output_W = self.add_weight(self.output_W_shape,
                                        initializer=initializers.get('glorot_uniform'),
                                        name='{}_W2'.format(self.name),
                                        regularizer=None,
                                        constraint=None)

        # self.output_b = self.add_weight((self.output_dim[0],),
        #                                 initializer='zero',
        #                                 name='{}_b2'.format(self.name),
        #                                 regularizer=None)

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (n_pasr_errors,samples,) + output_shape
        ndim = len(self.output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps,) + input_shape
        initial_state = K.sum(initial_state, axis=tuple(i for i in range(1, ndim + 2)))  # (samples,)
        if self.n_past_error != 1:
            initial_state = K.expand_dims(initial_state, 0)  # (1,samples)
            initial_state = K.tile(initial_state, [self.n_past_error, 1])  # (n_past_error,samples)
        for dim in self.output_dim:
            initial_state = K.expand_dims(initial_state)  # (n_past_error,samples,1)
            initial_state = K.tile(initial_state, [dim])  # (n_past_error, samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def call(self, x, mask=None):
        print "call"
        predict_targets = x[0]
        real_targets = x[1]

        init_state = self.get_initial_states(real_targets)
        ndim = predict_targets.ndim
        axes = [1, 0] + list(range(2, ndim))
        predict_targets = predict_targets.dimshuffle(axes)

        ndim = real_targets.ndim
        axes = [1, 0] + list(range(2, ndim))
        real_targets = real_targets.dimshuffle(axes)

        print type(init_state[0])
        print (init_state[0]).broadcastable

        if len(init_state) > 0:
            for i in range(1, 2 + len(self.output_dim)):
                init_state[0] = T.unbroadcast(init_state[0], i)

        # print (init_state[0]).broadcastable
        # exit(1)

        def _step(x_t, y_t, *errors):
            error = T.concatenate(errors, 1)
            o = K.conv2d(error, self.W, strides=(1, 1),
                         padding="same",
                         data_format="channels_first")
            o += K.reshape(self.b, (1, self.error_hidden_dim, 1, 1))

            o = activations.get("relu")(o)

            o = K.conv2d(o, self.output_W, strides=(1, 1),
                         padding="same",
                         data_format="channels_first")

            # o += K.reshape(self.output_b, (1, self.output_dim[0], 1, 1))

            o = activations.get("relu")(o)
            o += x_t
            return [o - y_t, o]

        init_state = init_state[0]
        # init_state = theano.shared(np.zeros((self.n_past_error, 926, 1, 48, 48), dtype=theano.config.floatX))
        [errors, os], _ = theano.scan(_step,
                                      sequences=[predict_targets, real_targets],
                                      outputs_info=[
                                          dict(initial=init_state, taps=[-i for i in range(self.n_past_error, 0, -1)]),
                                          None])
        last_os = os[-1]
        axes = [1, 0] + list(range(2, os.ndim))
        os = os.dimshuffle(axes)

        if self.return_sequences:
            return os
        else:
            return last_os

    def compute_output_shape(self, input_shape):
        print "get_output_shape_for", input_shape
        input_shape = input_shape[1]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1]) + self.output_dim
        else:
            output_shape = (input_shape[0],) + self.output_dim
        print "output_shape:", output_shape
        return output_shape


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


def _share_layer(shareIndex, shares, layer):
    assert len(shareIndex) == 1
    index = shareIndex[0]
    if (len(shares) <= index):
        shares.append(layer)
    share_layer = shares[index]
    shareIndex[0] = index + 1
    return share_layer


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False,
                  share=False, shareIndex=[0], shares=[]):
    def f(input):
        if bn:
            if share:
                input = _share_layer(shareIndex, shares, BatchNormalization(mode=0, axis=1))(input)

            else:
                input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        if share:
            o = _share_layer(shareIndex, shares,
                             Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                                           border_mode="same"))(input)
        else:
            o = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                              border_mode="same")(activation)

        return o

    return f


def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _residual_unit(nb_filter, subsample=(1, 1), share=False, shareIndex=[0], shares=[]):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3, shareIndex=shareIndex, share=share, shares=shares)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3, shareIndex=shareIndex, share=share, shares=shares)(residual)
        return _shortcut(input, residual)

    return f


def ResUnits(residual_unit, nb_filter, repetations=1, share=False, shareIndex=[0], shares=[]):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  subsample=init_subsample,
                                  share=share,
                                  shareIndex=shareIndex,
                                  shares=shares)(input)
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
    pkl = fname + '.preprocessing_speed.pkl'
    fn = "48_48_20_MaxSpeedFillingFixed_5"
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, \
        timestamp_train, timestamp_test, noConditionRegions, x_num, y_num, z_num = read_cache(fname, is_mmn,
                                                                                              pkl)
        print("load %s successfully" % fname)
    else:
        datapaths = [os.path.join(datapath, fn)]
        noConditionRegionsPath = os.path.join(datapath, "48_48_20_noSpeedRegion_0.05")
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, noConditionRegions, \
        x_num, y_num, z_num = Data.loadDataFromRaw(
                paths=datapaths, noSpeedRegionPath=noConditionRegionsPath, nb_flow=nb_flow, len_closeness=len_closeness,
                len_period=len_period, len_trend=len_trend
                , len_test=len_test, maxMinNormalization=is_mmn, preprocess_name=pkl,
                meta_data=hasExternal,
                meteorol_data=hasExternal,
                holiday_data=hasExternal, isComplete=False)

        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test, noConditionRegions, is_mmn, x_num, y_num,
                  nb_flow)
    z_num = nb_flow
    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::72]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model_train...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()

    X_train, Y_train = Data.getSequenceXY(X_train, Y_train, step)
    Y_train_final = Y_train[:, -1]
    X_test, Y_test = Data.getSequenceXY(X_test, Y_test, step)
    Y_test_final = Y_test[:, -1]
    X_train.append(Y_train)
    X_test.append(Y_test)

    timestamp_train = timestamp_train[step - 1:]
    timestamp_test = timestamp_test[step - 1:]

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
        Y_test_final = Y_test[:, -1]

    # print "X_test len:", len(X_test)
    # for x in X_test:
    #     print x.shape
    # print Y_test.shape
    # print z_num, x_num, y_num
    print "start build model_train"

    outputs = []
    inputs = []

    resUnit_share_layers = []
    resUnit_share_layers2 = []
    resUnit_share_layers3 = []
    shared_conv1 = Convolution2D(filters=64, kernel_size=(3, 3), padding="same")
    shared_conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")

    shared_conv3 = Convolution2D(filters=64, kernel_size=(3, 3), padding="same")
    shared_conv4 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")
    shared_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), padding="same")
    shared_conv6 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")

    shared_convLSTM_period = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode="same")
    shared_conv_period = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")

    shared_convLSTM_trend = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode="same")
    shared_conv_trend = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")

    shared_ilayers = []

    shared_embeding = Dense(output_dim=10)
    shared_embeding2 = Dense(output_dim=nb_flow * x_num * y_num)

    assert l < step
    for _ in range(step):
        main_outputs = []
        if len_closeness > 0:
            input = Input(shape=(nb_flow * len_closeness, x_num, y_num))
            inputs.append(input)
            # Conv1
            conv1 = shared_conv1(input)
            # [nb_residual_unit] Residual Units
            resUnit_share_index = [0]
            residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit, share=True,
                                       shareIndex=resUnit_share_index, shares=resUnit_share_layers)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = shared_conv2(activation)
            main_outputs.append(conv2)

            # input = Input(shape=(nb_flow * len_closeness, x_num, y_num))
            # inputs.append(input)
            # # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            # # act1 = Activation("relu")(conv1)
            # reshape = Reshape((len_closeness, nb_flow, x_num, y_num))(input)
            # convLSTM = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode="same")(reshape)
            # act2 = Activation("relu")(convLSTM)
            # conv2 = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(act2)
            # main_outputs.append(conv2)

        if len_period > 0:
            input = Input(shape=(nb_flow * len_period, x_num, y_num))
            inputs.append(input)
            # Conv1
            conv1 = shared_conv3(input)
            # [nb_residual_unit] Residual Units
            resUnit_share_index = [0]
            residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit, share=True,
                                       shareIndex=resUnit_share_index, shares=resUnit_share_layers2)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = shared_conv4(activation)
            main_outputs.append(conv2)
            # input = Input(shape=(nb_flow * len_period, x_num, y_num))
            # inputs.append(input)
            # # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            # # act1 = Activation("relu")(conv1)
            # input = Reshape((len_period, nb_flow, x_num, y_num))(input)
            # convLSTM = shared_convLSTM_period(input)
            # act2 = Activation("relu")(convLSTM)
            # conv2 = shared_conv_period(act2)
            # main_outputs.append(conv2)

        if len_trend > 0:
            input = Input(shape=(nb_flow * len_trend, x_num, y_num))
            inputs.append(input)
            # Conv1
            conv1 = shared_conv5(input)
            # [nb_residual_unit] Residual Units
            resUnit_share_index = [0]
            residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit, share=True,
                                       shareIndex=resUnit_share_index, shares=resUnit_share_layers3)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = shared_conv6(activation)
            main_outputs.append(conv2)
            # input = Input(shape=(nb_flow * len_trend, x_num, y_num))
            # inputs.append(input)
            # # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            # # act1 = Activation("relu")(conv1)
            # reshape = Reshape((len_trend, nb_flow, x_num, y_num))(input)
            # convLSTM = shared_convLSTM_trend(reshape)
            # act2 = Activation("relu")(convLSTM)
            # conv2 = shared_conv_trend(act2)
            # main_outputs.append(conv2)

        if len(main_outputs) == 1:
            main_output = main_outputs[0]
        else:
            new_outputs = []
            for index, output in enumerate(main_outputs):
                if (len(shared_ilayers) <= index):
                    shared_ilayers.append(iLayer())

                new_outputs.append(shared_ilayers[index](output))
            main_output = merge(new_outputs, mode='sum')

        if external_dim != None and external_dim > 0:
            # external input
            external_input = Input(shape=(external_dim,))
            inputs.append(external_input)
            embedding = shared_embeding(external_input)
            embedding = Activation('relu')(embedding)
            h1 = shared_embeding2(embedding)
            activation = Activation('relu')(h1)
            external_output = Reshape((nb_flow, x_num, y_num))(activation)
            main_output = merge([main_output, external_output], mode='sum')

        main_output = Activation('tanh')(main_output)
        outputs.append(main_output)

    main_output = merge(outputs, mode="concat", concat_axis=1)
    predict_sequence = Reshape((step, z_num, x_num, y_num))(main_output)

    input_targets = Input(shape=(step, z_num, x_num, y_num), name="input_targets")
    inputs.append(input_targets)
    main_output = eRNN(error_hidden_dim, (z_num, x_num, y_num), l, False)([predict_sequence, input_targets])

    model_train = Model(inputs=inputs, outputs=[predict_sequence, main_output])
    adam = Adam(lr=lr)
    model_train.compile(loss=['mse', 'mse'],
                        loss_weights=[0.2, 1],
                        optimizer=adam,
                        metrics=[metrics.rmse])
    # model_train.compile(loss=lambda y_true,y_preiod: K.mean(K.square(y_preiod - y_true), axis=-1), optimizer=adam, metrics=[metrics.rmse])
    # model_predict = Model(input=inputs, output=main_output)
    # model_predict.compile(optimizer=adam,loss="mse",metrics=metrics.rmse)
    model_train.summary()
    print "finish build model_train"

    hyperparams_name = 'testMyModel3_speed.c{}.p{}.t{}.resunit{}.lr{}.{}.{}'.format(
            len_closeness, len_period, len_trend, nb_residual_unit, lr,
            "External" if hasExternal else "noExternal",
            "MMN" if is_mmn else "noMMN")

    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_e_rnn_1_rmse', patience=4, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_e_rnn_1_rmse', verbose=0, save_best_only=True,
                                       mode='min',
                                       save_weights_only=True)

    print("\nelapsed time (compiling model_train): %.3f seconds\n" %
          (time.time() - ts))

    print('=' * 10)
    print("training model_train...")
    ts = time.time()

    history = model_train.fit(X_train, [Y_train, Y_train_final],
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              validation_split=0.1,
                              callbacks=[early_stopping, model_checkpoint],
                              verbose=1)

    model_train.save_weights(os.path.join(
            path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
            path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model_train that has the best loss on the valid set')
    ts = time.time()
    model_train.load_weights(fname_param)
    score = model_train.evaluate(X_train, [Y_train, Y_train_final], batch_size=Y_train.shape[0] // 48, verbose=0)

    if is_mmn:
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Train score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    score = model_train.evaluate(X_test, [Y_test, Y_test_final], batch_size=Y_test.shape[0] // 12, verbose=0)

    if is_mmn:
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    else:
        print('Test score: %.6f rmse (real): %.6f' %
              (score[0], score[1]))

    if not is_mmn:
        predict = model_train.predict(X_test)[1]
    else:
        predict = mmn.inverse_transform(model_train.predict(X_test)[1])
        Y_test_final = mmn.inverse_transform(Y_test_final)

    # predict = predict[:, -1]
    # Y_test = Y_test[:, -1]

    # print("predict", predict)
    # print("test", Y_test_final)
    rmse = round(Metric.RMSE(predict, Y_test_final, noConditionRegions), 6)
    np.save("./result/{}_predict_rmse{}".format(hyperparams_name, str(rmse)),
            np.stack([predict, Y_test_final], axis=0))
    print("RMSE:", rmse)
    # print("accuracy", Metric.accuracy(predict, Y_test, noConditionRegions))

    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    exit(1)


if __name__ == '__main__':
    main()
