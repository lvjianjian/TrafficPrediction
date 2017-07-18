#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-6-14, 15:42

@Description:

@Update Date: 17-6-14, 15:42
"""

import keras
from feature import Data
from conf import Paramater

if __name__ == '__main__':
    data, x_num, y_num, interval = Data.getData(Paramater.DATAPATH)
    keras.layers.convolutional.Conv2D(3, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                      dilation_rate=(1, 1), activation=None, use_bias=True,
                                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                      kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                      kernel_constraint=None, bias_constraint=None)
