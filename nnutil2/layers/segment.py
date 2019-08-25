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


import inspect

from tensorflow.python.keras.engine import network
import tensorflow as tf

class Segment(network.Network):
    """A collection of layers"""
    def __init__(self, layers=None, activation=None, **kwargs):
        super(Segment, self).__init__(**kwargs)
        assert isinstance(layers, list)

        self._segment_layers = layers
        self._segment_activation = tf.keras.activations.get(activation)
        self._segment_states = []

    def _compose(self, l, x, kwargs):
        sig = [p.name for p in inspect.signature(l.call).parameters.values()]
        args = {k: kwargs[k] for k in set(sig) & set(kwargs.keys())}
        y = l(x, **args)
        return y

    def get_config(self):
        config = {}
        return config

    def call(self, inputs, **kwargs):
        x = inputs
        self._segment_states.append(x)

        for l in self._segment_layers:
            x = self._compose(l, x, kwargs)
            self._segment_states.append(x)

        if self._segment_activation is not None:
            x = self._segment_activation(x)
            self._segment_states.append(x)

        return x

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for l in self._segment_layers:
            shape = l.compute_output_shape(shape)
        return shape

    @property
    def states(self):
        return self._segment_states
