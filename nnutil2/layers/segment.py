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

class Segment(network.Network):
    """A sequential collection of layers"""
    def __init__(self, layers: List[Layer] = [], activation=None, **kwargs):
        super(Segment, self).__init__(**kwargs)

        self._segment_layers = layers
        self._segment_activation = tf.keras.activations.get(activation)
        self._segment_states = []

    def get_config(self):
        config = {
            'layers': [ly.get_config() for ly in self._layers],
            'activation': self._segment_activation
        }

        base_config = super(Segment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        x = inputs
        self._segment_states.append(x)

        for l in self._segment_layers:
            layer_kwargs = kwargs_for(kwargs, l.call)
            x = l(x, **layer_kwargs)
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
    def flat_layers(self):
        layers = []

        def add_layers(ly):
            if isinstance(ly, Segment):
                for ly2 in ly.layers:
                    add_layers(ly2)
            else:
                layers.append(ly)

        add_layers(self)
        return layers

    @property
    def states(self):
        return self._segment_states
