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
import numpy as np

from .. import util
from .. import nest

def directional_derivative(f, x, v):
    """Compute the direcitonal derivative of f at x along v
    """

    tf.nest.assert_same_structure(x, v)
    dtype = nest.get_dtype(x)

    with tf.GradientTape() as tape:
        s = tf.constant(0, shape=(), dtype=dtype)
        tape.watch(s)

        x_sv = tf.nest.map_structure(lambda _x, _v: _x + s * _v, x, v)
        u = f(x_sv)
        u_flat = tf.nest.flatten(u)
        u_flat_vectors = [tf.reshape(ui, shape=(ui.shape.num_elements(),)) for ui in u_flat]

    u_s_flat = [tape.jacobian(ui, s) for ui in u_flat_vectors]

    u_s_flat = [tf.reshape(ui_s, shape=ui.shape) for ui_s, ui in zip(u_s_flat, u_flat)]
    u_s = tf.nest.pack_sequence_as(u, u_s_flat)

    return u_s
