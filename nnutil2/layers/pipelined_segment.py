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

from ..util import kwargs_for, as_shape
from .layer import Layer

class PipelinedSegment(network.Network):
    """A sequential collection of layers"""
    def __init__(self, layers: List[Layer]=[], activation=None, shape=None, dtype=None, **kwargs):
        super(PipelinedSegment, self).__init__(**kwargs)

        assert shape is not None
        assert dtype is not None

        nstages = len(layers)
        self._state_shape = (nstages,) + as_shape(shape)
        self._state_dtype = dtype
        self._segment_layers = layers
        self._segment_activation = tf.keras.activations.get(activation)

        self._input_state = None
        self._output_state = None

    def get_config(self):
        config = {
            'layers': [ly.get_config() for ly in self._layers],
            'activation': self._segment_activation,
            'shape': self._state_shape,
            'dtype': self._state_dtype,
        }

        base_config = super(PipelinedSegment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        nstages = self._state_shape[0]
        assert input_shape == self._state_shape[1:]

        self._pipeline_state = self.add_weight(
            "state",
            shape=self._state_shape,
            dtype=self._state_dtype,
            trainable=False,
            initializer=tf.keras.initializers.zeros(),
        )

    def call(self, inputs, **kwargs):
        nstages = len(self._segment_layers)

        in_state = tf.concat([
            tf.expand_dims(inputs, axis=0),
            self._pipeline_state[:-1,...]
        ], axis=0)

        self._input_state = in_state

        in_state_un = tf.split(in_state, nstages, axis=0)
        assert len(in_state_un) == nstages

        def apply_layer(l, x):
            layer_kwargs = kwargs_for(kwargs, l.call)
            y = l(x, **layer_kwargs)

            if self._segment_activation is not None:
                y = self._segment_activation(y)

            return y

        out_state_un = [apply_layer(l, x) for l, x in zip(self._segment_layers, in_state_un)]
        out_state = tf.concat(out_state_un, axis=0)
        out_state = self._pipeline_state.assign(out_state)
        self._output_state = out_state

        return out_state[-1,...]

    def compute_output_shape(self, input_shape):
        assert len(self._segment_layers) == input_shape[0]
        return input_shape

    @property
    def input_state(self):
        return self._input_state

    @property
    def output_state(self):
        return self._output_state

    @property
    def num_stages(self):
        return len(self._segment_layers)
