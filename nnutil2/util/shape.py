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

    elif isinstance(shape, (tf.Tensor, tf.TensorSpec)):
        return shape.shape

    elif isinstance(shape, list):
        if all([isinstance(x, int) for x in shape]):
            return tf.TensorShape(shape)

        else:
            return [as_shape(x) for x in shape]

    elif isinstance(shape, tuple):
        if all([isinstance(x, int) for x in shape]):
            return tf.TensorShape(shape)

        else:
            return tuple([as_shape(x) for x in shape])

    elif isinstance(shape, dict):
        return {k: as_shape(v) for k, v in shape.items()}

    else:
        raise Exception("Cannot handle input type: {}".format(type(shape)))

def rank(shape):
    shape = as_shape(shape)
    shape_flat = tf.nest.flatten(shape)
    rank_flat = [s.rank for s in shape_flat]

    rank = rank_flat[0]
    for r in rank_flat:
        assert r == rank

    return rank

def num_elements(shape):
    shape = as_shape(shape)
    shape_flat = tf.nest.flatten(shape)

    return sum([s.num_elements() for s in shape_flat])

def normalize_axis(shape, axis):
    """Returns the normalized shape dimension indices given by the axis specification"""
    shape = as_shape(shape)

    rank = shape.rank
    assert rank is not None

    if axis is None:
        return list(range(0, shape.rank))

    def normalize(d):
        return d if d >= 0 else rank + d

    if isinstance(axis, int):
        axis = normalize(axis)
        assert axis < rank
        return axis

    elif isinstance(axis, list):
        axis = [normalize(d) for d in axis]
        assert max(axis) < rank
        return axis

    else:
        raise Exception("Unhandled axis type: {}".format(type(axis)))

def complementary_axis(shape, axis):
    shape = as_shape(shape)

    rank = shape.rank
    assert rank is not None

    axis = normalize_axis(shape, axis)
    if isinstance(axis, int):
        axis = [axis]

    axis_c = [i for i in range(0, rank) if i not in set(axis)]
    return axis_c

def restrict_shape(shape, axis):
    """Restrict shape to the given axis"""
    shape = as_shape(shape)
    flat_shape = tf.nest.flatten(shape)
    flat_res_shape = [tf.TensorShape([s[i] for i in normalize_axis(s, axis)]) for s in flat_shape]

    res_shape = tf.nest.pack_sequence_as(shape, flat_res_shape)
    return res_shape

def reduce_shape(shape, axis):
    """Reduce shape along the given axis"""
    shape = as_shape(shape)
    flat_shape = tf.nest.flatten(shape)
    flat_res_shape = [tf.TensorShape([s[i] for i in complementary_axis(s, axis)]) for s in flat_shape]

    res_shape = tf.nest.pack_sequence_as(shape, flat_res_shape)
    return res_shape

def batch_shape(shape, inner_shape):
    """Return the batch_shape from shape, given inner_shape"""
    shape = as_shape(shape)
    inner_shape = as_shape(inner_shape)

    flat_shape = tf.nest.flatten(shape)
    flat_inner_shape = tf.nest.flatten(inner_shape)

    assert len(flat_shape) == len(flat_inner_shape)

    flat_batch_shape = []
    for sh, ish in zip(flat_shape, flat_inner_shape):
        assert sh.with_rank_at_least(ish.rank)

        batch_rank = sh.rank - ish.rank
        assert batch_rank >= 0

        assert sh[batch_rank:].is_compatible_with(ish)
        flat_batch_shape.append(sh[0:batch_rank])

    for bs in flat_batch_shape:
        assert bs == flat_batch_shape[0]

    return flat_batch_shape[0]

def infer_layer_shape(layer, input_shape, batch_rank=1):
    input_shape = as_shape(input_shape)
    extended_shape = tf.nest.map_structure(lambda s: tf.TensorShape(batch_rank * [1]) + s, input_shape)
    batched_output_shape = layer.compute_output_shape(extended_shape)

    output_shape = tf.nest.map_structure(lambda s: s[batch_rank:], batched_output_shape)
    return output_shape

def is_inner_compatible_with(shape0, shape1):
    """Check whether shape0 and shape1 are compatible on the inner dimensions.
       The higher rank must be compatible with the lower rank shape as tail.
    """
    shape0 = as_shape(shape0)
    shape1 = as_shape(shape1)

    try:
        tf.nest.assert_same_structure(shape0, shape1)
    except ValueError:
        return False

    flat_shape0 = tf.nest.flatten(shape0)
    flat_shape1 = tf.nest.flatten(shape1)

    def check(s0, s1):
        rank0 = s0.rank
        assert rank0 is not None

        rank1 = s1.rank
        assert rank1 is not None

        if rank0 < rank1:
            return s1[-rank0:].is_compatible_with(s0)
        else:
            return s0[-rank1:].is_compatible_with(s1)

    return all([check(s0, s1) for s0, s1 in zip(flat_shape0, flat_shape1)])

def is_outer_compatible_with(shape0, shape1):
    """Check whether shape0 contains shape1 are compatible on the inner dimensions.
       The higher rank must be compatible with the lower rank shape as tail.
    """
    shape0 = as_shape(shape0)
    shape1 = as_shape(shape1)

    try:
        tf.nest.assert_same_structure(shape0, shape1)
    except ValueError:
        return False

    flat_shape0 = tf.nest.flatten(shape0)
    flat_shape1 = tf.nest.flatten(shape1)

    def check(s0, s1):
        rank0 = s0.rank
        assert rank0 is not None

        rank1 = s1.rank
        assert rank1 is not None

        if rank0 < rank1:
            return s1[:rank0].is_compatible_with(s0)
        else:
            return s0[:rank1].is_compatible_with(s1)

    return all([check(s0, s1) for s0, s1 in zip(flat_shape0, flat_shape1)])

def outer_squeeze(x):
    shape = x.shape
    k = 0
    for i in range(0, shape.rank):
        if shape[i] == 1:
            k += 1
        else:
            break

    if k > 0:
        axis = list(range(0, k))
        return tf.squeeze(x, axis=axis)
    else:
        return x

def inner_squeeze(x):
    shape = x.shape
    k = 0
    for i in reversed(range(0, shape.rank)):
        if shape[i] == 1:
            k += 1
        else:
            break
    if k > 0:
        axis = list(range(shape.rank - k, shape.rank))
        return tf.squeeze(x, axis=axis)
    else:
        return x

def outer_broadcast(x, target):
    """Extends and broadcasts outer dimensions of x in order to match target
    """
    x = outer_squeeze(x)

    assert is_inner_compatible_with(x.shape, target.shape)

    rank_x = x.shape.rank
    rank_target = target.shape.rank
    padding_dim = rank_target - rank_x

    ones_shape = tf.TensorShape(padding_dim * (1,))
    new_shape = ones_shape.concatenate(x.shape)
    xnew = tf.reshape(x, shape=new_shape)

    # NOTE: We use dynamic shape here to accomodate dynamic batch shapes
    if isinstance(target, tf.TensorShape):
        target_shape = target
    elif isinstance(target, tf.Tensor):
        target_shape = tf.shape(target)
    else:
        raise Exception("Unhandled type: {}".format(type(target)))

    x_broadcast = tf.broadcast_to(xnew, shape=target_shape)

    return x_broadcast

def inner_broadcast(x, target):
    """Extends and broadcasts inner dimensions of x in order to match target
    """
    x = inner_squeeze(x)

    assert is_outer_compatible_with(x.shape, target.shape)

    rank_x = x.shape.rank
    rank_target = target.shape.rank
    padding_dim = rank_target - rank_x

    ones_shape = tf.TensorShape(padding_dim * (1,))
    new_shape = x.shape.concatenate(ones_shape)
    xnew = tf.reshape(x, shape=new_shape)

    # NOTE: We use dynamic shape here to accomodate dynamic batch shapes
    if isinstance(target, tf.TensorShape):
        target_shape = target
    elif isinstance(target, tf.Tensor):
        target_shape = tf.shape(target)
    else:
        raise Exception("Unhandled type: {}".format(type(target)))

    x_broadcast = tf.broadcast_to(xnew, shape=target_shape)

    return x_broadcast
