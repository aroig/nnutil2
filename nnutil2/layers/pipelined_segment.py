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

from .. import util
from .layer import Layer
from .. import nest

class PipelinedSegment(network.Network):
    """A sequential collection of layers"""
    def __init__(self, layers: List[Layer]=[], activation=None, dtype=tf.float32, **kwargs):
        super(PipelinedSegment, self).__init__(**kwargs)

        self._segment_layers = layers
        self._segment_activation = tf.keras.activations.get(activation)

        self._state_dtype = dtype

        self._input_state = None
        self._output_state = None

    def get_config(self):
        config = {
            'layers': [ly.get_config() for ly in self._layers],
            'activation': self._segment_activation,
        }

        base_config = super(PipelinedSegment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        nstages = len(self._segment_layers)

        self._state_shape = input_shape
        self._pipeline_state_shape = tf.nest.map_structure(lambda s: (nstages,) + s, self._state_shape)

        def _add_weight(shape):
            weight = self.add_weight(
                "state",
                shape=(nstages,) + shape,
                dtype = self._state_dtype,
                trainable=False,
                initializer=tf.keras.initializers.zeros(),
            )

            return weight

        self._pipeline_state = tf.nest.map_structure(_add_weight, self._state_shape)

    def pipeline_state(idx):
        tf.nest.map_structure(lambda x: x[idx:...], self._pipeline_state)

    def call(self, inputs, **kwargs):
        nstages = len(self._segment_layers)

        assert util.as_shape(inputs) == self._state_shape

        def apply_layer(l, x):
            layer_kwargs = util.kwargs_for(kwargs, l.call)
            y = l(x, **layer_kwargs)

            if self._segment_activation is not None:
                y = tf.nest.map_structure(self._segment_activation, y)

            return y

        def update_input_state(xi, state):
            xi = tf.expand_dims(xi, axis=0)
            new_state = tf.concat([xi, state[:-1,...]], axis=0)
            return new_state

        input_pipeline_state = tf.nest.map_structure(update_input_state, inputs, self._pipeline_state)

        output_pipeline_state = []
        for idx, l in enumerate(self._segment_layers):
            state_in = tf.nest.map_structure(lambda x: x[idx:idx+1,...], input_pipeline_state)
            state_out = apply_layer(l, state_in)
            output_pipeline_state.append(state_out)

        output_pipeline_state = tf.nest.map_structure(
            lambda *x: tf.concat(x, axis=0),
            *output_pipeline_state
        )

        assert util.as_shape(output_pipeline_state) == self._pipeline_state_shape

        self._pipeline_state = tf.nest.map_structure(
            lambda old, new: old.assign(new),
            self._pipeline_state, output_pipeline_state
        )

        self._input_state = inputs
        self._output_state = tf.nest.map_structure(lambda x: x[-1,...], self._pipeline_state)
        self._input_pipeline_state = input_pipeline_state

        return self._output_state

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def input_state(self):
        return self._input_state

    @property
    def output_state(self):
        return self._output_state

    @property
    def input_pipeline(self):
        return self._input_pipeline_state

    @property
    def output_pipeline(self):
        return self._pipeline_state

    @property
    def num_stages(self):
        return len(self._segment_layers)
