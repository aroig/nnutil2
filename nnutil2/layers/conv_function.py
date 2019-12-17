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

from .segment import Segment
from ..util import as_shape

class ConvFunction(Segment):
    """A collection of layers"""
    def __init__(self, input_shape=None, depth=None, residual=False, filters=1,
                 activation=None, layer_class=None, **kwargs):
        layers = []

        self._in_shape = as_shape(input_shape)
        self._filters = filters

        kernel_size = 3
        dimension = self._in_shape.rank - 1

        Layer = self._layer_class(self._in_shape, layer_class)

        nfeatures = int(self._in_shape[-1])
        conv_shape = np.array(list(self._in_shape[0:-1]))

        max_depth = int(np.round(np.log2(np.max(conv_shape))))
        if depth is not None:
            max_depth = max(max_depth, depth)

        self._depth = max_depth

        cur_shape = tf.TensorShape([1]) + self._in_shape
        for i in range(0, max_depth):
            alpha = 1 - (i / (max_depth - 1))
            expected_conv_shape = np.round(np.exp(alpha * np.log(conv_shape)))

            cur_conv_shape = np.array(list(cur_shape[1:-1]))
            strides = tuple([int(x) for x in np.round(cur_conv_shape / expected_conv_shape)])
            factor = np.prod(strides)

            if factor > 1:
                nfeatures *= factor

            conv_layer = Layer(
                filters=nfeatures,
                kernel_size=dimension * (kernel_size,),
                strides=strides,
                padding='same',
                activation=activation
            )

            cur_shape = conv_layer.compute_output_shape(cur_shape)

            layers.append(conv_layer)

        assert cur_shape == tf.TensorShape((dimension + 1) * [1] + [nfeatures])

        fc_layer = tf.keras.layers.Dense(
            units=self._filters,
            activation=tf.keras.activations.linear
        )

        layers.append(tf.keras.layers.Flatten())
        layers.append(fc_layer)

        super(ConvFunction, self).__init__(layers=layers, **kwargs)

    def _layer_class(self, shape, layer_class):
        if layer_class is not None:
            return layer_class

        if self._in_shape.rank == 1:
            return tf.keras.layers.Dense

        elif self._in_shape.rank == 2:
            return tf.keras.layers.Conv1D

        elif self._in_shape.rank == 3:
            return tf.keras.layers.Conv2D

        elif self._in_shape.rank == 4:
            return tf.keras.layers.Conv3D

        else:
            raise Exception("Input shape must have rank between 1 and 4")

    @property
    def depth(self):
        return self._depth

    @property
    def in_shape(self):
        return self._in_shape
