#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-20, 10:22

@Description:

@Update Date: 17-7-20, 10:22
"""
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
from deepst.models.STResNet import ResUnits
from deepst.models.iLayer import iLayer


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


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)

    return f


# def rounding(x):
#
#     return x


# class Rounding(Layer):
#     """Applies an activation function to an output.
#
#     # Arguments
#         activation: name of activation function to use
#             (see: [activations](../activations.md)),
#             or alternatively, a Theano or TensorFlow operation.
#
#     # Input shape
#         Arbitrary. Use the keyword argument `input_shape`
#         (tuple of integers, does not include the samples axis)
#         when using this layer as the first layer in a model.
#
#     # Output shape
#         Same shape as input.
#     """
#
#     def __init__(self, **kwargs):
#         self.supports_masking = True
#         self.activation = rounding
#         super(Rounding, self).__init__(**kwargs)
#
#     def call(self, x, mask=None):
#         return self.activation(x)
#
#     def get_config(self):
#         config = {'activation': self.activation.__name__}
#         base_config = super(Rounding, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3,
             isRegression=True):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
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
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = merge(new_outputs, mode='sum')

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = merge([main_output, external_output], mode='sum')
    else:
        print('external_dim:', external_dim)

    if isRegression:
        main_output = Activation('tanh')(main_output)
    else:
        main_output = Activation('relu')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model
