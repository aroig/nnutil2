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

def func_1(x):
    A = tf.linspace(0., 1., x.shape.num_elements())
    A = tf.reshape(A, shape=x.shape)
    return A * x


class LinalgLinearOperatorJacobian(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_linear_operator_jacobian_do_dense_1(self):
        shape = (16, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        x0 = tf.random.normal(shape=shape, dtype=tf.float32)

        def func(x):
            return A * x

        jac = nnu.linalg.LinearOperatorJacobian(func, x0, input_shape=(5,))
        self.assertAllClose(tf.linalg.diag(A), jac.to_dense())

    def test_linalg_linear_operator_jacobian_matmul_1(self):
        batch_size = 16

        x0 = tf.random.normal(shape=(batch_size, 5), dtype=tf.float32)
        v0 = tf.random.normal(shape=(batch_size, 5, 7), dtype=tf.float32)
        v1 = tf.random.normal(shape=(batch_size, 7, 5), dtype=tf.float32)
        jac = nnu.linalg.LinearOperatorJacobian(func_1, x0, input_shape=(5,))

        for adjoint in [True, False]:
            jac_v0_A = jac.matmul(v0, adjoint=adjoint, adjoint_arg=False)
            jac_v0_B = tf.linalg.matmul(jac.to_dense(), v0, adjoint_a=adjoint, adjoint_b=False)
            self.assertAllClose(jac_v0_A, jac_v0_B)

            jac_v1_A = jac.matmul(v1, adjoint=adjoint, adjoint_arg=True)
            jac_v1_B = tf.linalg.matmul(jac.to_dense(), v1, adjoint_a=adjoint, adjoint_b=True)
            self.assertAllClose(jac_v1_A, jac_v1_B)

    def test_linalg_linear_operator_jacobian_matvec_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        for adjoint in [True, False]:
            jac = nnu.linalg.LinearOperatorJacobian(func_1, x0, input_shape=(5,))
            jac_v0_A = jac.matvec(v0, adjoint=adjoint)
            jac_v0_B = tf.linalg.matvec(jac.to_dense(), v0, transpose_a=adjoint)
            self.assertAllClose(jac_v0_A, jac_v0_B)

    def test_linalg_linear_operator_jacobian_diag_part_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        jac = nnu.linalg.LinearOperatorJacobian(func_1, x0, input_shape=(5,))
        self.assertAllClose(jac.diag_part(), tf.linalg.diag_part(jac.to_dense()))

    def test_linalg_linear_operator_jacobian_trace_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        jac = nnu.linalg.LinearOperatorJacobian(func_1, x0, input_shape=(5,))
        self.assertAllClose(jac.trace(), tf.linalg.trace(jac.to_dense()))


if __name__ == '__main__':
    tf.test.main()
