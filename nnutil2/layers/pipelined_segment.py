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

from tensorflow.python.keras.engine import network
import tensorflow as tf

from ..util import kwargs_for
from .layer import Layer

class PipelinedSegment(network.Network):
    """A sequential collection of layers"""
    def __init__(self, layers: List[Layer]=[], activation=None, **kwargs):
        super(Segment, self).__init__(**kwargs)

        self._segment_layers = layers
        self._segment_activation = tf.keras.activations.get(activation)

    def get_config(self):
        config = {
            'layers': [ly.get_config() for ly in self._layers],
            'activation': self._segment_activation
        }

        base_config = super(Segment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        x0, xstate = inputs

        xstate_un = [x0] + tf.unstack(xstate)
        assert len(xstate_un) == len(self._segment_layers)

        def apply_layer(l, x):
            layer_kwargs = kwargs_for(kwargs, l.call)
            y = l(x, **layer_kwargs)

            if self._segment_activation is not None:
                y = self._segment_activation(y)

            return y

        ystate_un = [apply_layer(l, x) for l, x in zip(self._segment_layers, xstate_un)]

        ystate = tf.stack(ystate_un[:-1])
        yn = ystate_un[-1]

        return ystate, yn

    def compute_output_shape(self, input_shape):
        assert len(self._segment_layers) == input_shape[0]
        return input_shape

    @property
    def num_stages(self):
        return len(self._segment_layers)
