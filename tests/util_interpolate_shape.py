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

import unittest

import numpy as np
import tensorflow as tf
import nnutil2 as nnu

class UtilInterpolateShape(tf.test.TestCase):
    def test_interpolate_1(self):
        ts0 = tf.TensorShape([16, 16, 2])
        ts1 = tf.TensorShape([1, 1, 32])

        adaptors = list(nnu.util.interpolate_shape(ts0, ts1, 5))

        shapes = [sa.expected_shape for sa in adaptors]
        true_shapes = [
            tf.TensorShape([16, 16, 2]),
            tf.TensorShape([8, 8, 4]),
            tf.TensorShape([4, 4, 8]),
            tf.TensorShape([2, 2, 16]),
            tf.TensorShape([1, 1, 32])
        ]
        self.assertEqual(true_shapes, shapes)

        strides = [sa1.strides(sa0.expected_shape) for sa0, sa1 in zip(adaptors[:-1], adaptors[1:])]
        true_strides = 4 * [(2, 2)]
        self.assertEqual(true_strides, strides)

        dilations = [sa1.dilation_rate(sa0.expected_shape) for sa0, sa1 in zip(adaptors[:-1], adaptors[1:])]
        true_dilations = 4 * [(1, 1)]
        self.assertEqual(true_dilations, dilations)

        filters = [sa.filters for sa in adaptors]
        true_filters = [2, 4, 8, 16, 32]
        self.assertEqual(true_filters, filters)
