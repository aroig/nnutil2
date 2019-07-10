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
            feature1 = sess.run(next(it1))
            self.assertEqual({'a': 1}, feature1)

            it = iter(ds)
            feature = sess.run(next(it))
            self.assertEqual({'a': 1, 'b': 2, 'c': 3}, feature)


if __name__ == '__main__':
    tf.test.main()
