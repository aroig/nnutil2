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

def orthogonalize(v: tf.Tensor, A: tf.Tensor) -> tf.Tensor:
    """ Make v orthogonal to the span of the columns of A, by subtracting the projection to A.
    """
    batch_shape = v.shape[:-1]

    assert batch_shape == A.shape[:-2]
    assert A.shape[-2] == v.shape[-1]

    vexp = tf.expand_dims(v, axis=-2)
    norms = tf.reduce_sum(A * A, axis=-2)
    coeffs = tf.squeeze(tf.linalg.matmul(vexp, A), axis=-2)
    proj_v = tf.linalg.matvec(A, coeffs / norms)

    return v - proj_v
