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

class MathTraceMC(tf.test.TestCase):
    def setUp(self):
        pass

    def test_math_identity_mc_1(self):
        batch_shape = tf.TensorShape([2, 3])
        shape = tf.TensorShape([2, 2])

        with self.cached_session():
            z = nnu.math.identity_mc(shape, batch_rank=1, seed=42)
            self.assertEqual(z.shape, shape)

    def test_math_identity_mc_2(self):
        N = 1024
        shape = tf.TensorShape([N, 4])

        identity = tf.eye(4)

        with self.cached_session():
            z = nnu.math.identity_mc(shape, batch_rank=1, seed=42)
            approx_id = (1. / N) * tf.linalg.matmul(z, z, transpose_a=True)
            self.assertAllClose(approx_id, identity, atol=2e-1)

    def test_math_trace_mc(self):
        with self.cached_session():
            A = tf.constant([[[1, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=tf.float32)

            def f(x):
                return tf.linalg.matvec(A, x)

            tr = nnu.math.trace_mc(f, shape=(1024, 3), batch_rank=1)
            tr = tf.reduce_mean(tr, axis=0)
            self.assertEqual(tr.shape, tf.TensorShape([]))
            self.assertAllClose(tf.constant(9, dtype=tf.float32), tr, atol=2e-1)
