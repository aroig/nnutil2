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

from ..util import as_shape

class Normalization(Layer):
    """Normalization layer
    """
    def __init__(self, input_shape=None, mode=None, data_format='channel_last', **kwargs):
        assert input_shape is not None
        assert mode in ['batch', 'layer']
        assert data_format in ['channel_first', 'channel_last']

        self._in_shape = as_shape(input_shape)
        self._mode = mode

        if data_format == 'channels_first':
            norm_axis = 1
        else:
            norm_axis = -1

        if self._mode == 'batch':
            norm_layer = tf.keras.layers.BatchNormalization(
                input_shape=self._in_shape,
                axis=norm_axis,
                **kwargs
            )

        elif self._normalization == 'layer':
            norm_layer = tf.keras.layers.LayerNormalization(
                input_shape=self._in_shape,
                axis=norm_axis,
                **kwargs
            )

        else:
            raise Exception("Unknow normalization mode: {}".format(self._mode))

        self._layer = norm_layer

        super(Normalization, self).__init__()

    def get_config(self):
        config = {
            'input_shape': self._in_shape,
            'mode': self._mode,
            'layer': self._layer.get_config()
        }

        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        return self._layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self._layer.compute_output_shape(input_shape)
