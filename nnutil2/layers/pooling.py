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
from .identity import Identity

from ..util import as_shape

class Pooling(Layer):
    """Average Pooling layer wrapper that chooses the right kernel dimension
       based on input shape
    """
    def __init__(self, input_shape=None, reduction='average', *args, **kwargs):
        assert input_shape is not None
        self._in_shape = as_shape(input_shape)

        Layer = self._layer_class(input_shape, reduction=reduction)
        self._layer = Layer(*args, input_shape=input_shape, **kwargs)

        super(Pooling, self).__init__()

    def _layer_class(self, shape, reduction):
        if reduction == 'average':
            if self._in_shape.rank == 1:
                return Identity

            elif self._in_shape.rank == 2:
                return tf.keras.layers.AveragePooling1D

            elif self._in_shape.rank == 3:
                return tf.keras.layers.AveragePooling2D

            elif self._in_shape.rank == 4:
                return tf.keras.layers.AveragePooling3D

            else:
                raise Exception("Input shape must have rank between 1 and 4. Got {}".format(self._in_shape))

        elif reduction == 'max':
            if self._in_shape.rank == 1:
                return Identity

            elif self._in_shape.rank == 2:
                return tf.keras.layers.MaxPool1D

            elif self._in_shape.rank == 3:
                return tf.keras.layers.MaxPool2D

            elif self._in_shape.rank == 4:
                return tf.keras.layers.MaxPool3D

            else:
                raise Exception("Input shape must have rank between 1 and 4. Got {}".format(self._in_shape))
        else:
            raise Exception("Unknown reduction function {}".format(reduction))

    def get_config(self):
        config = {
            'input_shape': self._in_shape,
            'reduction': self._reduction
        }
        base_config = super(Pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        return self._layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self._layer.compute_output_shape(input_shape)
