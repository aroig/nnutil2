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

from .residual import Residual
from .segment import Segment
from ..util import as_shape, interpolate_shape

class ConvFunction(Segment):
    """A function defined by a segment of convolutional nets"""
    def __init__(self, input_shape=None, depth=None, output_shape=(1,), residual=False,
                 activation=None, layer_class=None, **kwargs):
        assert input_shape is not None

        layers = []

        self._in_shape = as_shape(input_shape)
        self._out_shape = as_shape(output_shape)
        self._layer_activation = tf.keras.activations.get(activation)
        self._residual = residual

        kernel_size = 3
        dimension = self._in_shape.rank - 1

        Layer = self._layer_class(self._in_shape, layer_class)

        max_depth = int(np.round(max([np.log2(x) for x in self._in_shape[0:-1]])))
        if depth is not None:
            max_depth = max(max_depth, depth)

        self._depth = max_depth

        nfeatures = int(self._in_shape[-1] * np.power(2, max_depth))

        shape0 = self._in_shape
        shape1 = tf.TensorShape(dimension * [1] + [nfeatures])

        cur_shape = tf.TensorShape([1]) + self._in_shape
        for sa in interpolate_shape(shape0, shape1, max_depth):
            conv_layer = Layer(
                filters=sa.filters,
                kernel_size=dimension * (kernel_size,),
                strides=sa.strides(cur_shape),
                dilation_rate=sa.dilation_rate(cur_shape),
                padding='same',
                activation=(tf.keras.activations.linear if self._residual else self._layer_activation)
            )

            if self._residual:
                conv_layer = Residual(layers=[conv_layer], activation=self._layer_activation)

            cur_shape = conv_layer.compute_output_shape(cur_shape)

            layers.append(conv_layer)

        assert cur_shape[1:] == shape1

        fc_layer = tf.keras.layers.Dense(
            units=self._out_shape.num_elements(),
            activation=tf.keras.activations.linear
        )

        layers.append(tf.keras.layers.Flatten())
        layers.append(fc_layer)
        layers.append(tf.keras.layers.Reshape(target_shape=self._out_shape))

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

    @property
    def out_shape(self):
        return self._out_shape
