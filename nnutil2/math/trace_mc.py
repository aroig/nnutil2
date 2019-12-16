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

def identity_mc(shape=None, batch_rank: int = 1, seed=None):
    """
       Produces a sample z such that

       E(z z^t) = Id
    """
    assert shape is not None
    shape = as_shape(shape)
    z = tf.random.normal(shape=shape, seed=seed)
    return z


def trace_mc(f, shape=None, batch_rank: int = 1, seed=None):
    """Compute an unbiased Monte-Carlo approximation of the trace of A

       A is given implicitly by f(v) = Av

       f must take a single tensor argument of given shape

       Take random variable z such that E(z z^t) = Id. Then

       tr A = E(<z, A z>) = E(<z, f(z)>)

    """
    assert shape is not None
    shape = as_shape(shape)
    assert shape.rank > batch_rank

    batch_shape = shape[:batch_rank]
    inner_shape = shape[batch_rank:]

    z = identity_mc(shape=shape, batch_rank=batch_rank, seed=seed)
    fz = f(z)

    axis = list(range(batch_rank, shape.rank))
    tr = tf.reduce_sum(z * fz, axis=axis)

    return tr