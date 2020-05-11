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

class LayersSegment(tf.test.TestCase):
    # def test_layers_segment_1(self):
    #     dense = tf.layers.Dense(units=4)
    #     testing_utils.layer_test(
    #         nnu.layers.Segment,
    #         kwargs={"layers": [dense]},
    #         input_shape=(2, 3))

    def test_layers_segment_2(self):
        dense = tf.keras.layers.Dense(units=4)
        segment = nnu.layers.Segment(layers=[dense])
        x = tf.random.normal(shape=(2, 3), dtype=tf.float32)
        y = segment(x)

        self.assertEqual(len(segment.layers), 1)
        self.assertEqual(len(segment.trainable_weights), 2)
        self.assertEqual(len(segment.states), 3)

        # self.assertEqual(segment.inputs[0].shape, (2, 3))

    def test_layers_segment_nested(self):
        identity0 = nnu.layers.Identity()
        identity1 = nnu.layers.Identity()

        segment = nnu.layers.Segment(layers=[identity0, identity1])
        x = {
            'a': tf.random.normal(shape=(2, 3), dtype=tf.float32),
            'b': tf.random.normal(shape=(4,), dtype=tf.float32),
        }

        y = segment(x)

        self.assertEqual(len(segment.layers), 2)
        self.assertEqual(len(segment.trainable_weights), 0)

        self.assertEqual(x['a'].shape, y['a'].shape)
        self.assertEqual(x['b'].shape, y['b'].shape)


if __name__ == '__main__':
    tf.test.main()
