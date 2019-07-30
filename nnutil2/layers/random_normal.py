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


import tensorflow as tf
import tensorflow.keras as ks

class RandomNormal(ks.layers.Layer):
    def __init__(self, shape=None, mean=0.0, stddev=1.0):
        assert shape is not None
        self._shape = shape

        self._mean = mean
        self._stddev = stddev

        super(RandomNormal, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.random.normal(shape=self._shape, mean=self._mean, stddev=self._stddev)

    def compute_output_shape(self, input_shape):
        return self._shape
