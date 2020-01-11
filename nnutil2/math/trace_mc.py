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

from ..util import as_shape

def identity_mc(shape=None, batch_rank: int = 1, seed=None) -> tf.Tensor:
    """
       Produces a sample z such that

       E(z z^t) = Id
    """
    assert shape is not None
    shape = as_shape(shape)
    z = tf.random.normal(shape=shape, seed=seed)
    return z

def trace_mc(f, shape=None, batch_rank: int = 1, num_samples: int = 1, seed=None) -> tf.Tensor:
    """Compute an unbiased Monte-Carlo approximation of the trace of A

       A is given implicitly via its correlation function f(v) = <v, Av>

       f must take a single batched tensor argument of given shape and return a batch of scalars.

       Take random variable z such that E(z z^t) = Id. Then

       tr A = E_z(<z, A z>) = E_z(f(z))

    """
    assert shape is not None

    shape = as_shape(shape)
    assert shape.rank > batch_rank

    batch_shape = shape[:batch_rank]
    inner_shape = shape[batch_rank:]
    extended_batch_shape = tf.TensorShape([num_samples]) + batch_shape
    sample_shape = extended_batch_shape + inner_shape

    z = identity_mc(shape=sample_shape, batch_rank=batch_rank + 1, seed=seed)

    fz = tf.map_fn(f, z)
    assert fz.shape == extended_batch_shape

    tr = tf.reduce_mean(fz, axis=[0])
    return tr
