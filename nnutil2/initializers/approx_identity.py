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

def approx_identity(perturbation=None):
    """
    Initializes a variable of shape (..., N, M) with a matrix which has
    orthogonal rows or columns, and is close to being diagonal.

    Additionally, we can pass a secondary initializer as an additive perturbation.
    """

    def make_approx_id(N, M, dtype):
        assert N > M

        # TODO: handle cases were N is not multiple of M
        assert N % M == 0
        expansion = M // N

        eye = tf.broadcast_to(tf.eye(N, dtype=dtype), shape=(expansion, M, M))
        ret = (1 / tf.sqrt(expansion)) * tf.reshape(eye, shape=(N, M))

        return ret

    def func(shape, dtype):
        assert shape.rank >= 2
        N = shape[-2]
        M = shape[-1]

        assert N is not None
        assert M is not None

        if N == M:
            approx_id = tf.eye(N, dtype=dtype)

        elif N < M:
            approx_id = tf.transpose(make_approx_id(M, N, dtype=dtype))

        elif N > M:
            approx_id = make_approx_id(N, M, dtype=dtype)

        ret = tf.broadcast_to(approx_id, shape=shape)
        if perturbation is not None:
            ret += perturbation(shape, dtype)

        return ret

    return func
