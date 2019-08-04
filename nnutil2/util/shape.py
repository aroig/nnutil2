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

def as_shape(shape):
    """Return shape as a tf.TensorShape object"""
    if isinstance(shape, tf.TensorShape):
        return shape
    else:
        return tf.TensorShape(shape)

def shape_dimensions(shape, axis):
    """Returns the shape dimensions given by the axis specification"""
    shape = as_shape(shape)
    if axis is None:
        return list(range(0, shape.rank))

    rank = shape.rank
    axis = [d if d >= 0 else rank - d for d in axis]

    assert max(axis) < shape.rank

    return axis

def restrict_shape(shape, axis):
    """Restrict shape to the given axis"""
    axis = shape_dimensions(shape, axis)
    return tf.TensorShape([shape[i] for i in axis])

def reduce_shape(shape, axis):
    """Reduce shape along the given axis"""
    axis = shape_dimensions(shape, axis)
    shape = as_shape(shape)

    return tf.TensorShape([shape[i] for i in range(0, shape.rank) if i not in set(axis)])
