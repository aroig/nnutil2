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

class LayersConv(tf.test.TestCase):
    def test_layer_full_conv_1D(self):
        with self.cached_session() as sess:
            shape = (5, 1)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.Conv(
                input_shape=shape,
                filters=8,
                kernel_size=(3,),
                activation=tf.keras.activations.relu
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 3, 8]), y.shape)

    def test_layer_full_conv_2D(self):
        with self.cached_session() as sess:
            shape = (5, 5, 1)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.Conv(
                input_shape=shape,
                filters=8,
                kernel_size=(3, 3),
                activation=tf.keras.activations.relu
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 3, 3, 8]), y.shape)

    def test_layer_full_conv_3D(self):
        with self.cached_session() as sess:
            shape = (5, 5, 5, 1)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.Conv(
                input_shape=shape,
                filters=8,
                kernel_size=(3, 3, 3),
                activation=tf.keras.activations.relu
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 3, 3, 3, 8]), y.shape)

    def test_layer_separable_conv_1D(self):
        with self.cached_session() as sess:
            shape = (5, 1)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.Conv(
                input_shape=shape,
                mode='separable',
                filters=8,
                kernel_size=(3,),
                activation=tf.keras.activations.relu
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 3, 8]), y.shape)

    def test_layer_separable_conv_2D(self):
        with self.cached_session() as sess:
            shape = (5, 5, 1)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.Conv(
                input_shape=shape,
                mode='separable',
                filters=8,
                kernel_size=(3, 3),
                activation=tf.keras.activations.relu
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 3, 3, 8]), y.shape)

    def test_layer_depthwise_conv_2D(self):
        with self.cached_session() as sess:
            shape = (5, 5, 8)
            x = tf.random.normal(shape=(4,) + shape, dtype=tf.float32)

            layer = nnu.layers.Conv(
                input_shape=shape,
                mode='depthwise',
                kernel_size=(3, 3),
                activation=tf.keras.activations.relu
            )

            y = layer(x)
            self.assertEqual(tf.TensorShape([4, 3, 3, 8]), y.shape)


if __name__ == '__main__':
    tf.test.main()
