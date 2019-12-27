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
from .global_pooling import GlobalPooling
from .segment import Segment

from ..util import as_shape

class SqueezeExcitation(Layer):
    """Squeeze and Excitation block
    """
    def __init__(self, input_shape=None, contraction=0.5, activation=None, **kwargs):
        assert input_shape is not None
        self._in_shape = as_shape(input_shape)
        self._contraction = contraction

        nchannels = int(self._in_shape[-1])
        self._nchannels = nchannels

        self._squeeze = GlobalPooling(
            reduction='average',
            input_shape=self._in_shape
        )

        rank = self._in_shape.rank
        target_shape = tf.TensorShape((rank - 1) * [1] + [nchannels])

        self._excitation = Segment(layers=[
            tf.keras.layers.Dense(units=int(nchannels * contraction), activation=activation),
            tf.keras.layers.Dense(units=nchannels, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Reshape(target_shape=target_shape)
        ])

        super(SqueezeExcitation, self).__init__()

    def get_config(self):
        config = {
            'input_shape': self._in_shape,
            'contraction': self._contraction
        }
        base_config = super(SqueezeExcitation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        z = self._squeeze(inputs)
        s = self._excitation(z)

        y = s * inputs
        return y

    def compute_output_shape(self, input_shape):
        return input_shape
