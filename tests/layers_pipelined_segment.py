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
        shape=(8,)

        segment = nnu.layers.PipelinedSegment(layers=[stage0, stage1], dtype=tf.float32)
        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        x1 = tf.random.normal(shape=shape, dtype=tf.float32)

        y0 = segment(x0)
        y1 = segment(x1)

        self.assertAllClose(y1, 2 * x0)
        self.assertAllClose(segment.input_pipeline[0,:], segment.output_pipeline[0,:])
        self.assertAllClose(2 * segment.input_pipeline[1,:], segment.output_pipeline[1,:])

    def test_layers_pipelined_segment_2(self):
        stage0 = tf.keras.layers.Dense(8)
        stage1 = tf.keras.layers.Dense(8)
        shape = (8,)

        segment = nnu.layers.PipelinedSegment(layers=[stage0, stage1], dtype=tf.float32)
        x0 = tf.random.normal(shape=shape, dtype=tf.float32)

        y0 = segment(x0)

        self.assertEqual(4, len(segment.trainable_variables))


if __name__ == '__main__':
    tf.test.main()
