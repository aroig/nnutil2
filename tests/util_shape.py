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
        pass

    def test_complementary_axis(self):
        pass

    def test_restrict_shape(self):
        pass

    def test_reduce_shape(self):
        pass

    def test_batch_shape(self):
        pass

    def test_is_inner_compatible_with(self):
        pass

if __name__ == '__main__':
    tf.test.main()
