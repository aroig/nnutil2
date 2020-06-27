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

class Stacked(Layer):
    """A sequential collection of layers"""
    def __init__(self, layers: List[Layer]=[], activation=None, **kwargs):
        super(Stacked, self).__init__(**kwargs)

        self._segment_layers = layers
        self._segment_activation = tf.keras.activations.get(activation)

        self._input_state = None
        self._output_state = None

    def get_config(self):
        config = {
            'layers': [ly.get_config() for ly in self._layers],
            'activation': self._segment_activation,
        }

        base_config = super(Stacked, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        def apply_layer(l, x):
            layer_kwargs = util.kwargs_for(kwargs, l.call)
            y = l(x, **layer_kwargs)

            if self._segment_activation is not None:
                y = tf.nest.map_structure(self._segment_activation, y)

            return y

        output_state_un = []
        for idx, l in enumerate(self._segment_layers):
            state_in = tf.nest.map_structure(lambda x: x[idx:idx+1,...], inputs)
            state_out = apply_layer(l, state_in)
            output_state_un.append(state_out)

        output_state = tf.nest.map_structure(
            lambda *x: tf.concat(x, axis=0),
            *output_state_un
        )

        self._input_state = inputs
        self._output_state = output_state

        return self._output_state

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def input_state(self):
        return self._input_state

    @property
    def output_state(self):
        return self._output_state
