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

class DataMerge(tf.test.TestCase):
    def test_dataset_merge_1(self):
        tf.random.set_seed(42)

        ds1 = tf.data.Dataset.from_tensors({
            'a': tf.constant(1, dtype=tf.int32)
        })

        ds2 = tf.data.Dataset.from_tensors({
            'b': tf.constant(2, dtype=tf.int32),
            'c': tf.constant(3, dtype=tf.int32)
        })

        ds = nnu.data.Merge([ds1, ds2])

        with self.cached_session() as sess:
            it1 = iter(ds1)
            feature1 = next(it1)
            self.assertEqual(set(feature1.keys()), set(['a']))

            self.assertEqual(1, feature1['a'].numpy())

            it = iter(ds)
            feature = next(it)
            self.assertEqual(set(feature.keys()), set(['a', 'b', 'c']))

            self.assertEqual(1, feature['a'].numpy())
            self.assertEqual(2, feature['b'].numpy())
            self.assertEqual(3, feature['c'].numpy())


if __name__ == '__main__':
    tf.test.main()
