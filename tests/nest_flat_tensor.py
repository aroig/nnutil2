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

class NestFlatTensor(tf.test.TestCase):
    def test_flatten_vector(self):
        tf.random.set_seed(42)

        x = {
            'a': tf.constant([[1,2], [3,4]], dtype=tf.int32),
            'b': tf.constant([10, 20], dtype=tf.int32),
        }

        shape = {
            'a': tf.TensorShape([2]),
            'b': tf.TensorShape([]),
        }

        xflat = tf.constant([[1,2,10], [3, 4, 20]], dtype=tf.int32)
        self.assertAllEqual(xflat, nnu.nest.flatten_vector(x, shape))

    def test_unflatten_vector(self):
        x = {
            'a': tf.constant([[1,2], [3,4]], dtype=tf.int32),
            'b': tf.constant([10, 20], dtype=tf.int32),
        }

        shape = {
            'a': tf.TensorShape([2]),
            'b': tf.TensorShape([]),
        }

        batch_shape = tf.TensorShape([2])

        xflat = tf.constant([[1,2,10], [3, 4, 20]], dtype=tf.int32)

        x_computed = nnu.nest.unflatten_vector(xflat, shape, batch_shape)
        self.assertAllEqual(x['a'], x_computed['a'])
        self.assertAllEqual(x['b'], x_computed['b'])

if __name__ == '__main__':
    tf.test.main()
