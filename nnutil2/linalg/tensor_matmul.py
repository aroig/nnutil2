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

from ..util import normalize_axis

def tensor_matmul(mat=None, x=None, axis=None, transpose=False, adjoint=False):
    assert mat is not None
    assert x is not None
    assert axis is not None

    batch_shape = mat.shape[:-2]

    shape = x.shape
    axis = normalize_axis(shape, axis)
    axis_dim = shape[axis]

    assert batch_shape.rank <= axis

    tail_shape = shape[axis+1:]
    tail_size = tail_shape.num_elements()

    ones = (axis - batch_shape.rank) * (1,)
    mat = tf.reshape(mat, batch_shape + ones + mat.shape[-2:])

    x_flat = tf.reshape(x, shape=shape[:axis+1] + (tail_size,))
    y_flat = tf.linalg.matmul(mat, x_flat, transpose_a=transpose, adjoint_a=adjoint)
    y = tf.reshape(y_flat, shape=y_flat.shape[:axis+1] + shape[axis+1:])

    return y
