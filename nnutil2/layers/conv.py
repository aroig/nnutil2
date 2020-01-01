#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2019, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

import numpy as np
import tensorflow as tf

from .layer import Layer
from .segment import Segment
from .residual import Residual
from .normalization import Normalization

from ..util import as_shape

class Conv(Layer):
    """Convolutional layer wrapper that chooses the right kernel dimension
       based on input shape
    """
    def __init__(self, input_shape=None, mode='full', activation=None,
                 residual=False, normalization=None, data_format='channels_last',
                 *args, **kwargs):
        assert input_shape is not None
        assert mode in ['full', 'depthwise', 'separable']
        assert data_format in ['channels_first', 'channels_last']

        self._in_shape = as_shape(input_shape)
        self._mode = mode
        self._residual = residual
        self._normalization = normalization
        self._data_format = data_format

        Layer = self._layer_class(input_shape, mode)

        conv = Layer(
            *args,
            input_shape=input_shape,
            data_format=data_format,
            activation=tf.keras.activations.linear,
            **kwargs)

        layers = [conv]

        if self._normalization is not None:
            cur_shape = conv.compute_output_shape(tf.TensorShape([1]) + input_shape)[1:]
            norm = Normalization(
                input_shape=cur_shape,
                data_format=self._data_format,
                mode=self._normalization)

            layers.append(norm)

        if self._residual:
            self._conv_layers = Residual(layers=layers, activation=activation)
        else:
            self._conv_layers = Segment(layers=layers, activation=activation)

        super(Conv, self).__init__()

    def _layer_class(self, shape, mode):
        if mode == 'full':
            if self._in_shape.rank == 1:
                return tf.keras.layers.Dense

            elif self._in_shape.rank == 2:
                return tf.keras.layers.Conv1D

            elif self._in_shape.rank == 3:
                return tf.keras.layers.Conv2D

            elif self._in_shape.rank == 4:
                return tf.keras.layers.Conv3D

            else:
                raise Exception("Input shape must have rank between 1 and 4 for convolutions. Got {}".format(self._in_shape))

        elif mode == 'separable':
            if self._in_shape.rank == 1:
                return tf.keras.layers.Dense

            elif self._in_shape.rank == 2:
                return tf.keras.layers.SeparableConv1D

            elif self._in_shape.rank == 3:
                return tf.keras.layers.SeparableConv2D

            else:
                raise Exception("Input shape must have rank between 1 and 3 for separable convolutions. Got {}".format(self._in_shape))

        elif mode == 'depthwise':
            if self._in_shape.rank == 3:
                return tf.keras.layers.DepthwiseConv2D

            else:
                raise Exception("Input shape must have rank 3 for depthwise convolutions. Got {}".format(self._in_shape))

        else:
            raise Exception("Unrecognized convolution mode. Got {}".format(mode))

    def get_config(self):
        config = {
            'input_shape': self._in_shape,
            'mode': self._mode,
            'residual': self._residual,
            'normalization': self._normalization,
            'data_format': self._data_format,
            'layer': self._layer.get_config()
        }

        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        return self._conv_layers(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self._conv_layers.compute_output_shape(input_shape)
