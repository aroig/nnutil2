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

class LinalgSymmetricLanczos(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_symmetric_lanczos_1(self):
        N = 32
        M = 8
        batch_size = 16

        shape = (batch_size, N, N)

        A = nnu.linalg.symmetrize(tf.random.normal(shape=shape), axis=[-1, -2])

        T, V = nnu.linalg.symmetric_lanczos(
            lambda v: tf.linalg.matvec(A, v),
            lambda : nnu.random.uniform_unit_vector(shape=shape[:-1]),
            size=M,
            orthogonalize_step=True
        )

        Vt = tf.linalg.matrix_transpose(V)

        identity = tf.eye(M, M, batch_shape=(batch_size,))
        self.assertAllClose(identity, Vt @ V, atol=1e-5)

        self.assertAllClose(T, Vt @ A @ V, atol=1e-5)

if __name__ == '__main__':
    tf.test.main()
