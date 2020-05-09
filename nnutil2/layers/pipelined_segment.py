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

    def call(self, inputs, **kwargs):
        nstages = len(self._segment_layers)

        # TODO: assert compatible

        def apply_layer(l, x):
            layer_kwargs = kwargs_for(kwargs, l.call)
            y = l(x, **layer_kwargs)

            if self._segment_activation is not None:
                y = self._segment_activation(y)

            return y

        flat_pipeline_state = tf.nest.flatten(self._pipeline_state)
        flat_inputs = tf.nest.flatten(inputs)

        flat_output_pipeline = []
        flat_output_state = []
        flat_input_pipeline = []
        for x0, state in zip(flat_inputs, flat_pipeline_state):
            state_un = tf.split(state, nstages, axis=0)
            in_pipeline_un = [tf.expand_dims(x0, axis=0)] + state_un[:-1]
            out_pipeline_un = [apply_layer(l, xi) for l, xi in zip(self._segment_layers, in_pipeline_un)]

            in_pipeline = tf.concat(in_pipeline_un, axis=0)
            out_pipeline = tf.concat(out_pipeline_un, axis=0)
            out_pipeline = state.assign(out_pipeline)
            output_state = out_pipeline[-1,...]

            flat_input_pipeline.append(in_pipeline)
            flat_output_pipeline.append(out_pipeline)
            flat_output_state.append(output_state)

        output_pipeline_state = tf.nest.pack_sequence_as(self._pipeline_state, flat_output_pipeline)
        output_state = tf.nest.pack_sequence_as(self._pipeline_state, flat_output_state)
        input_pipeline_state = tf.nest.pack_sequence_as(self._pipeline_state, flat_input_pipeline)

        self._pipeline_state = output_pipeline_state
        self._input_state = inputs
        self._output_state = output_state
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
