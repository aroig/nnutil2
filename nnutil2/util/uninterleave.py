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

def uninterleave(x, num_shards, axis=0):
    axis = normalize_axis(x.shape, axis)
    axis_dim = x.shape[axis]

    assert axis_dim % num_shards == 0

    newshape = x.shape[:axis] + (axis_dim // num_shards, num_shards) + x.shape[axis+1:]

    xnew = tf.reshape(x, shape=newshape)

    shard_shape = x.shape.as_list()
    shard_shape[axis] = axis_dim // num_shards

    shards = []
    for i in range(0, num_shards):
        beg = (x.shape.rank + 1) * [0]
        beg[axis+1] = i
        size = x.shape[:axis] + (axis_dim // num_shards, 1) + x.shape[axis+1:]
        xshard = tf.slice(xnew, beg, size)
        xshard = tf.reshape(xshard, shape=shard_shape)
        shards.append(xshard)

    return shards
