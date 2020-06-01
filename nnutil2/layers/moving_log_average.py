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

class MovingLogAverage(Layer):
    """An exponential moving average"""
    def __init__(self, decay=0.999, dtype=tf.float32, **kwargs):
        super(MovingLogAverage, self).__init__(**kwargs)
        self._decay = decay
        self._dtype = dtype

        self._state_shape = None
        self._state = None

    def get_config(self):
        config = {
            'decay': self._decay,
        }

        base_config = super(MovingLogAverage, self).get_config()
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
                initializer=tf.keras.initializers.zeros(),
                dtype=self._dtype,
                trainable=False
            )

        self._state = tf.nest.map_structure(add_weight, input_shape)

    def call(self, inputs, **kwargs):
        """
        S_i = log ( δ * exp( S_i-1 ) + (1 - δ) * exp (xi) )
            = S_i-1 + log δ + log( 1 + exp (xi - S_i-1 + log (1 - δ) - log(δ)) )
        """
        bias_correction = self._bias_correction.assign_add(tf.math.log(self._decay))

        A = tf.math.log(1 - self._decay) - tf.math.log(self._decay)

        def update(state, x):
            delta = tf.math.log(self._decay) + tf.math.softplus(x - state + A)
            newval = state.assign_add(delta)

            newval_corr = newval - tf.math.log(1 - tf.math.exp(bias_correction))
            return newval_corr

        y = tf.nest.map_structure(update, self._state, inputs)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def average(self):
        return self._state
