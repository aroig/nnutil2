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

from tensorflow.python.keras import testing_utils

class LayersConfFunction(tf.test.TestCase):
    def test_layer_shape_1D(self):
        with self.cached_session() as sess:
            shape = (5, 1)
            x = tf.random.normal(shape=(3,) + shape, dtype=tf.float32)
            func = nnu.layers.ConvFunction(shape=shape, depth=4, filters=2, activation=tf.keras.activations.relu)

            y = func(x)
            self.assertEqual(tf.TensorShape([3, 2]), y.shape)
            self.assertEqual(4, func.depth)

    def test_layer_shape_2D(self):
        with self.cached_session() as sess:
            shape = (5, 5, 1)
            x = tf.random.normal(shape=(3,) + shape, dtype=tf.float32)
            func = nnu.layers.ConvFunction(shape=shape, depth=4, filters=2, activation=tf.keras.activations.relu)

            y = func(x)
            self.assertEqual(tf.TensorShape([3, 2]), y.shape)
            self.assertEqual(4, func.depth)

    def test_layer_shape_3D(self):
        with self.cached_session() as sess:
            shape = (5, 5, 5, 1)
            x = tf.random.normal(shape=(3,) + shape, dtype=tf.float32)
            func = nnu.layers.ConvFunction(shape=shape, depth=4, filters=2, activation=tf.keras.activations.relu)

            y = func(x)
            self.assertEqual(tf.TensorShape([3, 2]), y.shape)
            self.assertEqual(4, func.depth)


if __name__ == '__main__':
    tf.test.main()
