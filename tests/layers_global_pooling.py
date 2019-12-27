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

class LayersGlobalPooling(tf.test.TestCase):
    def test_layer_average_pooling_1D(self):
        with self.cached_session() as sess:
            shape = (6, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.GlobalPooling(
                input_shape=shape,
                reduction='average'
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 8]), y.shape)

    def test_layer_average_pooling_2D(self):
        with self.cached_session() as sess:
            shape = (6, 6, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.GlobalPooling(
                input_shape=shape,
                reduction='average'
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 8]), y.shape)

    def test_layer_average_pooling_3D(self):
        with self.cached_session() as sess:
            shape = (6, 6, 6, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.GlobalPooling(
                input_shape=shape,
                reduction='average'
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 8]), y.shape)

    def test_layer_max_pooling_1D(self):
        with self.cached_session() as sess:
            shape = (6, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.GlobalPooling(
                input_shape=shape,
                reduction='max'
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 8]), y.shape)

    def test_layer_max_pooling_2D(self):
        with self.cached_session() as sess:
            shape = (6, 6, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.GlobalPooling(
                input_shape=shape,
                reduction='max'
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 8]), y.shape)

    def test_layer_max_pooling_3D(self):
        with self.cached_session() as sess:
            shape = (6, 6, 6, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.GlobalPooling(
                input_shape=shape,
                reduction='max'
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 8]), y.shape)




if __name__ == '__main__':
    tf.test.main()
