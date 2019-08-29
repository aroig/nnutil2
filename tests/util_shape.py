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

class UtilShape(tf.test.TestCase):
    def test_as_shape(self):
        tf.random.set_seed(42)

        ts0 = tf.TensorShape([1, 2, 3])
        self.assertEqual(ts0, nnu.util.as_shape([1, 2, 3]))
        self.assertEqual(ts0, nnu.util.as_shape(tuple([1, 2, 3])))
        self.assertEqual(ts0, nnu.util.as_shape(tf.zeros(shape=[1, 2, 3])))

    def test_normalize_axis(self):
        axis0 = [-2, 1]
        shape0 = tf.TensorShape([10, 20, 30, 40])
        axisnorm0 = [2, 1]
        self.assertEqual(axisnorm0, nnu.util.normalize_axis(shape0, axis0))

        axis1 = [1, 23]
        shape1 = tf.TensorShape([10, 20])
        with self.assertRaises(AssertionError):
            nnu.util.normalize_axis(shape1, axis1)

    def test_complementary_axis(self):
        axis0 = [-2, 1]
        shape0 = tf.TensorShape([10, 20, 30, 40])
        axiscomp0 = [0, 3]
        self.assertEqual(axiscomp0, nnu.util.complementary_axis(shape0, axis0))

    def test_restrict_shape(self):
        axis0 = [-2, 1]
        shape0 = tf.TensorShape([10, 20, 30, 40])
        shaperes0 = tf.TensorShape([30, 20])
        self.assertEqual(shaperes0, nnu.util.restrict_shape(shape0, axis0))

    def test_reduce_shape(self):
        axis0 = [-2, 1]
        shape0 = tf.TensorShape([10, 20, 30, 40])
        shapered0 = tf.TensorShape([10, 40])
        self.assertEqual(shapered0, nnu.util.reduce_shape(shape0, axis0))

    def test_batch_shape(self):
        shape0 = tf.TensorShape([10, 20, 30, 40])
        inner_shape0 = tf.TensorShape([30, 40])
        batch_shape0 = tf.TensorShape([10, 20])
        self.assertEqual(batch_shape0, nnu.util.batch_shape(shape0, inner_shape0))

    def test_is_inner_compatible_with(self):
        shape0 = tf.TensorShape([10, 20, 30, 40])
        shape1 = tf.TensorShape([30, 40])
        self.assertTrue(nnu.util.is_inner_compatible_with(shape0, shape1))
        self.assertTrue(nnu.util.is_inner_compatible_with(shape1, shape0))

    def test_outer_broadcast(self):
        tgt = tf.zeros(shape=[3, 2, 1])
        x = tf.constant([[20], [30]], shape=(2, 1), dtype=tf.int32)
        xbr = nnu.util.outer_broadcast(x, tgt)

        xtrue = tf.constant([[[20], [30]], [[20], [30]], [[20], [30]]], shape=(3, 2, 1), dtype=tf.int32)
        self.assertAllEqual(xtrue, xbr)


if __name__ == '__main__':
    tf.test.main()
