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
from .conv import Conv
from .segment import Segment
from .residual import Residual
from .normalization import Normalization

from ..util import as_shape

class Bottleneck(Layer):
    """ Mobilenet style bottleneck block
    """
    def __init__(self, input_shape=None, filters=None, depth_multiplier=1, activation=None,
                 residual=False, normalization=None, data_format='channels_last', **kwargs):
        assert input_shape is not None
        assert filters is not None
        assert data_format in ['channels_first', 'channels_last']

        self._in_shape = as_shape(input_shape)
        self._filters = filters
        self._depth_multiplier = depth_multiplier
        self._residual = residual
        self._normalization = normalization
        self._data_format=data_format

        inner_channels = depth_multiplier * filters

        shape0 = tf.TensorShape([1]) + self._in_shape
        conv0 = Conv(input_shape=shape0[1:],
                     filters=inner_channels,
                     data_format=data_format,
                     activation=activation,
                     **kwargs)

        shape1 = conv0.compute_output_shape(shape0)
        conv1 = Conv(input_shape=shape1[1:],
                     mode='depthwise',
                     depth_multiplier=1,
                     data_format=data_format,
                     activation=activation,
                     **kwargs)

        shape2 = conv1.compute_output_shape(shape1)
        conv2 = Conv(input_shape=shape2[1:],
                     filters=filters,
                     data_format=data_format,
                     activation=None,
                     **kwargs)

        layers = [conv0, conv1, conv2]

        if self._normalization is not None:
            norm = Normalization(
                input_shape=shape2,
                data_format=self._data_format,
                mode=self._normalization)

            layers.append(norm)

        if self._residual:
            self._bottleneck_layers = Residual(layers=layers, activation=activation)
        else:
            self._bottleneck_layers = Segment(layers=layers, activation=activation)

        super(Bottleneck, self).__init__()

    def get_config(self):
        config = {
            'input_shape': self._in_shape,
            'filters': self._filters,
            'depth_multiplier': self._depth_multiplier,
            'residual': self._residual,
            'normalization': self._normalization,
            'data_format': self._data_format,
        }

        base_config = super(Bottleneck, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        y = self._bottleneck_layers(inputs, **kwargs)
        return y

    def compute_output_shape(self, input_shape):
        return self._layers.compute_output_shape(input_shape)
