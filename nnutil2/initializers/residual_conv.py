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

from ..util import as_shape
from .approx_identity import approx_identity


def residual_conv(kernel_rank=None, perturbation=None):
    approx_id = approx_identity()

    def func(shape, dtype):
        shape = as_shape(shape)

        if kernel_rank is None:
            rank = shape.rank - 2
        else:
            rank = kernel_rank

        kernel_shape = shape[-rank - 2:-2]
        kernel_size = kernel_shape.num_elements()

        onehot_kernel = tf.one_hot(kernel_size // 2, kernel_size)
        onehot_kernel = tf.reshape(onehot_kernel, shape=kernel_shape + (1, 1))

        channel_id = approx_id(shape=shape, dtype=dtype)

        ret = tf.reshape(onehot_kernel * channel_id, shape=shape)
        if perturbation is not None:
            ret += perturbation(shape, dtype)

        return ret

    return func
