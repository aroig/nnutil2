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

import unittest

import numpy as np
import tensorflow as tf

import nnutil2 as nnu

class LinalgTensorMatmul(tf.test.TestCase):
    def test_tensor_matmul_1(self):
        m = tf.random.normal(shape=(3, 4, 5), dtype=tf.float32)
        x = tf.random.normal(shape=(3, 5, 7), dtype=tf.float32)

        y = nnu.linalg.tensor_matmul(m, x, axis=-2)
        self.assertAllClose(y, m @ x)

    def test_tensor_matmul_2(self):
        m = tf.random.normal(shape=(3, 4, 5), dtype=tf.float32)
        x = tf.random.normal(shape=(3, 5, 7, 3), dtype=tf.float32)

        y = nnu.linalg.tensor_matmul(m, x, axis=-3)
        self.assertEqual(y.shape, tf.TensorShape([3, 4, 7, 3]))


if __name__ == '__main__':
    tf.test.main()
