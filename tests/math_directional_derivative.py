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

class MathDirectionalDerivative(tf.test.TestCase):
    def test_math_directional_derivative_1(self):
        v0 = tf.constant([1,0], dtype=tf.float32)
        x0 = tf.constant([1,1], dtype=tf.float32)
        f = lambda x: tf.square(x)

        f_v0 = nnu.math.directional_derivative(f, x0, v0)
        f_v0_exp = tf.constant([2, 0], dtype=tf.float32)

        self.assertAllClose(f_v0_exp, f_v0)

    def test_math_directional_derivative_2(self):
        v0 = {'a': tf.constant([1,0], dtype=tf.float32), 'b': tf.constant([5, 5], dtype=tf.float32)}
        x0 = {'a': tf.constant([1,1], dtype=tf.float32), 'b': tf.constant([2, -3], dtype=tf.float32)}
        f = lambda x: {'c': tf.square(x['a']) + 3 * tf.square(x['b'])}

        f_v0 = nnu.math.directional_derivative(f, x0, v0)
        f_v0_exp = {'c': tf.constant([62, -90], dtype=tf.float32)}

        self.assertAllClose(f_v0_exp['c'], f_v0['c'])

if __name__ == '__main__':
    tf.test.main()
