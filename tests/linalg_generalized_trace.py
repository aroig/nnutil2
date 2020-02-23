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

import tensorflow as tf

import nnutil2 as nnu

class LinalgGeneralizedTrace(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_trace_log_1(self):
        N = 64
        M = 8
        sample_size = 128
        batch_size = 4

        shape = (batch_size, N, N)

        A0 = tf.eye(N, N, batch_shape=(batch_size,))
        A1 = nnu.linalg.symmetrize(tf.random.normal(shape=shape))

        A = A0 + 0.2 * A1

        g = lambda x: tf.math.log(x)

        tr = nnu.linalg.generalized_trace(A, g)

        tr_mc = nnu.linalg.generalized_trace_mc(
            lambda v: tf.linalg.matvec(A, v),
            g,
            shape=shape[:-1],
            lanczos_size=M,
            num_samples=sample_size
        )

        self.assertAllClose(tr, tr_mc, atol=1e-3)

if __name__ == '__main__':
    tf.test.main()
