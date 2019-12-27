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

import numpy as np
import tensorflow as tf


class ShapeAdaptor:
    def __init__(self, expected_shape):
        self._expected_shape = expected_shape
        self._rank = expected_shape.size

    @property
    def rank(self):
        return self._rank

    @property
    def expected_shape(self):
        return tf.TensorShape([int(x) for x in np.round(self._expected_shape)])

    @property
    def filters(self):
        return int(self.expected_shape[-1])

    def strides(self, input_shape):
        assert input_shape.rank >= self._rank

        input_shape = list(input_shape[-self._rank:-1])
        output_shape = list(self._expected_shape[:-1])
        return tuple([int(max(1, np.round(x/y))) for x, y in zip(input_shape, output_shape)])

    def dilation_rate(self, input_shape):
        assert input_shape.rank >= self._rank

        input_shape = list(input_shape[-self._rank:-1])
        output_shape = list(self._expected_shape[:-1])
        return tuple([int(max(1, np.round(y/x))) for x, y in zip(input_shape, output_shape)])

def interpolate_shape(input_shape, output_shape, steps):
    assert steps > 1

    shape0 = np.array(list(input_shape), dtype=np.float32)
    shape1 = np.array(list(output_shape), dtype=np.float32)
    shape_delta = np.log(shape1) - np.log(shape0)

    # shape_t = round exp (log(shape0) + t (log(shape1) - log(shape0)))
    for i in range(0, steps):
        alpha = i / (steps - 1)
        expected_shape = np.exp(np.log(shape0) + alpha * (shape_delta))
        yield ShapeAdaptor(expected_shape)
