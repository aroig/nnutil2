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

from .distribution import distribution

def layer_sizes(layers=None, name='layer/sizes', step=None, description=None):
    assert layers is not None

    layer_sizes = []

    for ly in layers:
        size = sum([w.shape.num_elements() for w in ly.variables])
        layer_sizes.append(size)

    layer_sizes = tf.constant(layer_sizes, dtype=tf.float32)

    return distribution(layer_sizes,
                        name=name,
                        step=step,
                        description=description)
