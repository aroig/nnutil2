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


from functools import reduce

import tensorflow as tf
import tensorflow.keras as ks

from .layer import Layer

class Feature(Layer):
    def __init__(self, features, axis=None, flatten=False):
        super(Feature, self).__init__()

        self._axis = axis
        if self._axis is not None and self._axis >= 0:
            self._axis += 1

        self._flatten = flatten

        self._features = features

    def build(self, input_shape):
        pass

    def call(self, inputs, dtype=None):
        if dtype is None:
            dtype = tf.float32

        if self._axis is None and len(self._features) == 1:
            feature = inputs[self._features[0]]
        elif self._axis is not None:
            feature = tf.stack([tf.cast(inputs[k], dtype=dtype) for k in self._features], axis=self._axis)
        else:
            raise Exception("Need an axis to concatenate features")

        if self._flatten:
            size = reduce(lambda x, y: x * y, feature.shape[1:], 1)
            feature = tf.reshape(feature, shape=(-1, size))

        return feature

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((None, len(self._features)))
