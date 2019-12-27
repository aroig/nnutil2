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

class Conv(Layer):
    """Convolutional layer wrapper that chooses the right kernel dimension
       based on input shape
    """
    def __init__(self, input_shape=None, mode='full', *args, **kwargs):
        assert input_shape is not None
        self._in_shape = as_shape(input_shape)

        self._mode = mode

        Layer = self._layer_class(input_shape, mode)
        self._layer = Layer(*args, input_shape=input_shape, **kwargs)

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
                raise Exception("Input shape must have rank between 1 and 4 for convolutions")

        elif mode == 'separable':
            if self._in_shape.rank == 1:
                return tf.keras.layers.Dense

            elif self._in_shape.rank == 2:
                return tf.keras.layers.SeparableConv1D

            elif self._in_shape.rank == 3:
                return tf.keras.layers.SeparableConv2D

            else:
                raise Exception("Input shape must have rank between 1 and 3 for separable convolutions")

        elif mode == 'depthwise':
            if self._in_shape.rank == 2:
                return tf.keras.layers.DepthwiseConv2D

            else:
                raise Exception("Input shape must have rank 3 for depthwise convolutions")

        else:
            raise Exception("Unrecognized convolution mode".format(mode))

    def get_config(self):
        return self._layer.get_config()

    def call(self, inputs, **kwargs):
        return self._layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self._layer.compute_output_shape(input_shape)
