#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2020, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

import unittest

import numpy as np
import tensorflow as tf

import nnutil2 as nnu

class LinalgLinearOperatorAntisymmetrize(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_linear_operator_antisymmetrize_to_dense_1(self):
        shape = (16, 5, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        A_alt = 0.5 * (A - tf.linalg.adjoint(A))

        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        A_op_alt = nnu.linalg.LinearOperatorAntisymmetrize(A_op)

        self.assertAllClose(A_alt, A_op_alt.to_dense())

    def test_linalg_linear_operator_antisymmetrize_matmul_1(self):
        shape = (16, 5, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        A_alt = 0.5 * (A - tf.linalg.adjoint(A))

        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        A_op_alt = nnu.linalg.LinearOperatorAntisymmetrize(A_op)

        v0 = tf.random.normal(shape=(16, 5, 7), dtype=tf.float32)

        A_v0_A = tf.matmul(A_alt, v0)
        A_v0_B = A_op_alt.matmul(v0)

        self.assertAllClose(A_v0_A, A_v0_B)

    def test_linalg_linear_operator_antisymmetrize_diag_part_1(self):
        shape = (16, 5, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        A_op_alt = nnu.linalg.LinearOperatorAntisymmetrize(A_op)
        zero = tf.zeros(dtype=tf.float32, shape=(16, 5))

        self.assertAllClose(A_op_alt.diag_part(), zero)


if __name__ == '__main__':
    tf.test.main()
