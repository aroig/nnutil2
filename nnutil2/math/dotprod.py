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

def dotprod(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Produces dot product of last dimensions
    """
    assert x.shape == y.shape
    x_mat = tf.expand_dims(x, axis=-2)
    xy = tf.linalg.matvec(x_mat, y)
    xy = tf.squeeze(xy, axis=-1)
    return xy
