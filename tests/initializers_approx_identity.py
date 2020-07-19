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

class InitializersApproxIdentity(tf.test.TestCase):
    def test_approx_identity_1(self):
        approx_id = nnu.initializers.approx_identity()

        x = approx_id(shape=(3, 3), dtype=tf.float32)
        self.assertAllClose(x, tf.eye(3))

    def test_approx_identity_2(self):
        approx_id = nnu.initializers.approx_identity()

        x = approx_id(shape=(2, 3, 6), dtype=tf.float32)
        self.assertEqual(x.shape, tf.TensorShape([2, 3, 6]))

    def test_approx_identity_3(self):
        approx_id = nnu.initializers.approx_identity()

        x = approx_id(shape=(2, 6, 3), dtype=tf.float32)
        self.assertEqual(x.shape, tf.TensorShape([2, 6, 3]))


if __name__ == '__main__':
    tf.test.main()
