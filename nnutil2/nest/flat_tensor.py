#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2020, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.


import tensorflow as tf

from .. import util

def flatten_vector(value, inner_structure):
    """
    Returns nested structure as a flat tensor (batch, inner_size)
    """
    batch_shape = util.batch_shape(value, inner_structure)
    batch_size = batch_shape.num_elements()

    flat_values = [tf.reshape(x, shape=(batch_size, util.as_shape(s).num_elements()))
                   for x, s in zip(tf.nest.flatten(value), tf.nest.flatten(inner_structure))]

    if len(flat_values) == 1:
        flat_vector = flat_values[0]
    else:
        flat_vector = tf.concat(flat_values, axis=-1)

    return flat_vector


def unflatten_vector(value, inner_structure, batch_shape):
    """
    Unflattens a batch of vectors (batch, inner_size) into a nested structure
    """
    batch_shape = util.as_shape(batch_shape)
    inner_shape = util.as_shape(inner_structure)
    flat_inner_shape = tf.nest.flatten(inner_shape)

    assert value.shape.rank == 2

    batch_size = value.shape[0]
    inner_size = value.shape[1]

    assert batch_size == batch_shape.num_elements()
    assert inner_size == util.num_elements(inner_shape)

    if len(flat_inner_shape) == 1:
        flat_values = [value]
    else:
        flat_values = tf.split(value, num_or_size_splits=[s.num_elements() for s in flat_inner_shape], axis=1)

    unflat_values = [tf.reshape(x, shape=batch_shape + s) for x, s in zip(flat_values, flat_inner_shape)]
    unflat_values = tf.nest.pack_sequence_as(inner_structure, unflat_values)

    return unflat_values
