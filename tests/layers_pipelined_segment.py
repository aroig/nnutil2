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

class LayersPipelinedSegment(tf.test.TestCase):
    def test_layers_pipelined_segment_1(self):
        stage0 = tf.keras.layers.Lambda(lambda x: x)
        stage1 = tf.keras.layers.Lambda(lambda x: 2*x)

        segment = nnu.layers.Segment(layers=[stage0, stage1])
        x = tf.random.normal(shape=(2, 16), dtype=tf.float32)
        y = segment(x)

        self.assertAllClose(x[0,:], y[0,:])
        self.assertAllClose(2 * x[1,:], y[1,:])

    def test_layers_pipelined_segment_1(self):
        stage0 = tf.keras.layers.Dense(4)
        stage1 = tf.keras.layers.Dense(4)

        segment = nnu.layers.Segment(layers=[stage0, stage1])
        x = tf.random.normal(shape=(2, 16), dtype=tf.float32)
        y = segment(x)

        self.assertEqual(4, len(segment.trainable_variables))


if __name__ == '__main__':
    tf.test.main()
