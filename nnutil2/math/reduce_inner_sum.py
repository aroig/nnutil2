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
from ..util import is_inner_compatible_with


def reduce_inner_sum(input_tensor, inner_shape):
    """Reduces the inner dimensions matching shape
    """
    input_shape = input_tensor.shape
    assert is_inner_compatible_with(input_shape, inner_shape)

    axis = list(range(input_shape.rank - inner_shape.rank, input_shape.rank))
    return tf.math.reduce_sum(input_tensor, axis=axis)
