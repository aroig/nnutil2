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

class MathDotprod(tf.test.TestCase):
    def setUp(self):
        pass

    def test_math_dotprod(self):
        x = tf.constant([[1, 2, 3], [0, 1, 0]], dtype=tf.float32)
        y = tf.constant([[1, 1, 1], [0, 0, 1]], dtype=tf.float32)

        xy0 = nnu.math.dotprod(x, y)
        xy1 = tf.constant([6, 0], dtype=tf.float32)

        with self.cached_session():
            self.assertAllClose(xy1, xy0)
