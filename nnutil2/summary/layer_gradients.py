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

def layer_gradients(layers=None, optimizer=None, loss=None,
                    name='layer/gradients', step=None, description=None):
    assert layers is not None
    assert optimizer is not None
    assert loss is not None

    layer_gradients = []

    def norm_or_zero(x):
        if x is None:
            return 0
        else:
            return tf.linalg.norm(x)

    for ly in layers:
        grads = [norm_or_zero(g) for w in ly.variables
                 for g in optimizer.get_gradients(loss, w)]

        layer_gradients.append(tf.constant(0, dtype=tf.float32) + sum(grads))

    layer_gradients = tf.stack(layer_gradients)

    return distribution(layer_gradients,
                        name=name,
                        step=step,
                        description=description)
