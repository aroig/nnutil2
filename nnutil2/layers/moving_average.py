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

from typing import List

import tensorflow as tf

from .. import util
from .layer import Layer
from .. import nest

class MovingAverage(Layer):
    """An exponential moving average"""
    def __init__(self, decay=0.999, dtype=tf.float32, **kwargs):
        super(MovingAverage, self).__init__(**kwargs)
        self._decay = decay
        self._dtype = dtype

        self._state_shape = None
        self._state = None

    def get_config(self):
        config = {
            'decay': self._decay,
        }

        base_config = super(MovingAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self._state_shape = input_shape

        self._bias_correction = self.add_weight(
            shape=(),
            dtype=self._dtype,
            initializer=tf.keras.initializers.zeros(),
            trainable = False,
        )

        def add_weight(s):
            return self.add_weight(
                shape=s,
                dtype=self._dtype,
                initializer=tf.keras.initializers.zeros(),
                trainable=False,
            )

        self._state = tf.nest.map_structure(add_weight, input_shape)

    def call(self, inputs, **kwargs):
        bias_correction = self._bias_correction.assign_add(tf.math.log(self._decay))

        def update(state, x):
            newval = self._decay * state + (1 - self._decay) * x
            newval = state.assign(newval)

            newval_corr = newval / (1 - tf.math.exp(bias_correction))
            return newval_corr

        y = tf.nest.map_structure(update, self._state, inputs)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def average(self):
        return self._state
