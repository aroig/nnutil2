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

def uniform_unit_vector(shape):
    """produces a uniformly distributed unit vector
    """

    x = tf.random.normal(shape=shape)
    xnorm, norm = tf.linalg.normalize(x, axis=-1)
    return xnorm
