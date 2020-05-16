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

from .shape import normalize_axis

def interleave(xn, axis=0):
    shape = xn[0].shape

    assert all([x.shape == shape for x in xn])

    axis = normalize_axis(shape, axis)
    axis_dim = shape[axis]

    xn_exp = [tf.expand_dims(x, axis=axis+1) for x in xn]

    x_interleaved = tf.concat(xn_exp, axis=axis+1)

    new_shape = shape.as_list()
    new_shape[axis] = len(xn) * shape[axis]
    x_interleaved = tf.reshape(x_interleaved, shape=new_shape)

    return x_interleaved
