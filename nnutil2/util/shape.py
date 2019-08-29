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
    elif isinstance(shape, tf.Tensor):
        return shape.shape
    elif isinstance(shape, list) or isinstance(shape, tuple):
        return tf.TensorShape(shape)
    else:
        raise Exception("Cannot handle input type: {}".format(type(shape)))

def normalize_axis(shape, axis):
    """Returns the normalized shape dimension indices given by the axis specification"""
    shape = as_shape(shape)

    rank = shape.rank
    assert rank is not None

    if axis is None:
        return list(range(0, shape.rank))

    axis = [d if d >= 0 else rank + d for d in axis]
    assert max(axis) < rank

    return axis

def complementary_axis(shape, axis):
    shape = as_shape(shape)

    rank = shape.rank
    assert rank is not None

    axis = normalize_axis(shape, axis)
    axis_c = [i for i in range(0, rank) if i not in set(axis)]
    return axis_c

def restrict_shape(shape, axis):
    """Restrict shape to the given axis"""
    shape = as_shape(shape)
    axis = normalize_axis(shape, axis)
    return tf.TensorShape([shape[i] for i in axis])

def reduce_shape(shape, axis):
    """Reduce shape along the given axis"""
    shape = as_shape(shape)
    axis_c = complementary_axis(shape, axis)
    return restrict_shape(shape, axis_c)

def batch_shape(shape, inner_shape):
    """Return the batch_shape from shape, given inner_shape"""
    shape = as_shape(shape)
    inner_shape = as_shape(inner_shape)

    assert shape.with_rank_at_least(inner_shape.rank)

    batch_rank = shape.rank - inner_shape.rank
    assert batch_rank >= 0

    assert shape[batch_rank:].is_compatible_with(inner_shape)
    return shape[0:batch_rank]

def is_inner_compatible_with(shape0, shape1):
    """Check whether shape0 contains shape1 are compatible on the inner dimensions.
       The higher rank must be compatible with the lower rank shape as tail.
    """
    shape0 = as_shape(shape0)
    shape1 = as_shape(shape1)

    rank0 = shape0.rank
    assert rank0 is not None

    rank1 = shape1.rank
    assert rank1 is not None

    if rank0 < rank1:
        return shape1[-rank0:].is_compatible_with(shape0)
    else:
        return shape0[-rank1:].is_compatible_with(shape1)

def outer_broadcast(x, target):
    """Extends and broadcasts outer dimensions of x in order to match target
    """
    assert is_inner_compatible_with(x.shape, target.shape)

    rank_x = x.shape.rank
    rank_target = target.shape.rank
    padding_dim = rank_target - rank_x

    new_shape = tf.TensorShape(padding_dim * (1,)).concatenate(x.shape)
    xnew = tf.reshape(x, shape=new_shape)

    # NOTE: We use dynamic shape here to accomodate dynamic batch shapes
    xnew = tf.broadcast_to(x, shape=tf.shape(target))

    return xnew
