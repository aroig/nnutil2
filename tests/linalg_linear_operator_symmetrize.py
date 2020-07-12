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

class LinalgLinearOperatorSymmetrize(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_linear_operator_symmetrize_to_dense_1(self):
        shape = (16, 5, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        A_sym = 0.5 * (A + tf.linalg.adjoint(A))

        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        A_op_sym = nnu.linalg.LinearOperatorSymmetrize(A_op)

        self.assertAllClose(A_sym, A_op_sym.to_dense())

    def test_linalg_linear_operator_symmetrize_matmul_1(self):
        shape = (16, 5, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        A_sym = 0.5 * (A + tf.linalg.adjoint(A))

        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        A_op_sym = nnu.linalg.LinearOperatorSymmetrize(A_op)

        v0 = tf.random.normal(shape=(16, 5, 7), dtype=tf.float32)

        A_v0_A = tf.matmul(A_sym, v0)
        A_v0_B = A_op_sym.matmul(v0)

        self.assertAllClose(A_v0_A, A_v0_B)

    def test_linalg_linear_operator_symmetrize_diag_part_1(self):
        shape = (16, 5, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        A_op_sym = nnu.linalg.LinearOperatorSymmetrize(A_op)

        self.assertAllClose(A_op_sym.diag_part(), tf.linalg.diag_part(A))


if __name__ == '__main__':
    tf.test.main()
