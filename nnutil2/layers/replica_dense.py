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

import tensorflow as tf

from .layer import Layer

from ..util import as_shape, normalize_axis, complementary_axis, num_elements

class ReplicaDense(Layer):
    """
    Dense layer replicated along a dimension
    """
    def __init__(self, nfilters=None, replica_axes=None, contraction_axes=None, dtype=tf.float32, *args, **kwargs):
        assert nfilters is not None
        assert replica_axes is not None
        assert contraction_axes is not None

        assert len(nfilters) == len(contraction_axes)

        self._dtype = dtype
        self._nfilters = as_shape(nfilters)
        self._replica_axes = replica_axes
        self._contraction_axes = contraction_axes

        super(ReplicaDense, self).__init__()

    def build(self, input_shape):
        contraction_axes = normalize_axis(input_shape, self._contraction_axes)
        replica_axes = normalize_axis(input_shape, self._replica_axes)

        assert len(set(contraction_axes) & set(replica_axes)) == 0

        contraction_dim = num_elements(input_shape, contraction_axes)
        replica_dim = num_elements(input_shape, replica_axes)
        nfilters_dim = num_elements(self._nfilters)

        self._weights = self.add_weight(
            name="weights",
            shape=(replica_dim, nfilters_dim, contraction_dim),
            dtype=self._dtype
        )

        self._bias = self.add_weight(
            name="bias",
            shape=(replica_dim, nfilters_dim, 1),
            dtype=self._dtype
        )

    def get_config(self):
        config = {
            "nfilters": self._nfilters,
            "replica_axes": self._replica_axes,
            "contraction_axes": self._contraction_axes,
            "dtype": self._dtype
        }

        base_config = super(ReplicaDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape
        contraction_axes = normalize_axis(input_shape, self._contraction_axes)
        replica_axes = normalize_axis(input_shape, self._replica_axes)
        complementary_axes = complementary_axis(input_shape, contraction_axes + replica_axes)

        # TODO: generalize this to arbitrary interleaving of dimensions
        assert max(replica_axes) < min(contraction_axes + complementary_axes)
        assert max(contraction_axes) < min(complementary_axes)

        flat_shape = tf.TensorShape([
            num_elements(input_shape, replica_axes),
            num_elements(input_shape, contraction_axes),
            num_elements(input_shape, complementary_axes)
        ])

        assert flat_shape.num_elements() == input_shape.num_elements()

        x = tf.reshape(inputs, shape=flat_shape)
        y = tf.linalg.matmul(self._weights, x) + self._bias

        output_shape = self.compute_output_shape(input_shape)
        assert output_shape.num_elements() == y.shape.num_elements()

        y = tf.reshape(y, shape=output_shape)

        return y

    def compute_output_shape(self, input_shape):
        contraction_axes = normalize_axis(input_shape, self._contraction_axes)
        output_shape = input_shape.as_list()

        for i, a in enumerate(contraction_axes):
            output_shape[a] = self._nfilters[i]

        return tf.TensorShape(output_shape)
