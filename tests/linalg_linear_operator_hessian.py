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

    return 0.5 * tf.reduce_sum(A * tf.square(x), axis=-1)


class LinalgLinearOperatorHessian(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_linear_operator_hessian_do_dense_1(self):
        shape = (16, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        x0 = tf.random.normal(shape=shape, dtype=tf.float32)

        def func(x):
            return 0.5 * tf.reduce_sum(A * tf.square(x), axis=-1)

        Hess = nnu.linalg.LinearOperatorHessian(func, x0)

        self.assertAllClose(tf.linalg.diag(A), Hess.to_dense())

    def test_linalg_linear_operator_hessian_directional_deriviative_1(self):
        shape = (16, 5)

        A = tf.random.normal(shape=shape, dtype=tf.float32)
        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        def func(x):
            return 0.5 * tf.reduce_sum(A * tf.square(x), axis=-1)

        Hess = nnu.linalg.LinearOperatorHessian(func, x0)

        self.assertAllClose(Hess.directional_derivative(v0), tf.reduce_sum(A * x0 * v0, axis=-1))

    def test_linalg_linear_operator_hessian_matmul_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape + (5,), dtype=tf.float32)

        Hess = nnu.linalg.LinearOperatorHessian(func_1, x0)

        self.assertAllClose(Hess.matmul(v0), tf.linalg.matmul(Hess.to_dense(), v0))

    def test_linalg_linear_operator_hessian_matvec_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        Hess = nnu.linalg.LinearOperatorHessian(func_1, x0)

        self.assertAllClose(Hess.matvec(v0), tf.linalg.matvec(Hess.to_dense(), v0))

    def test_linalg_linear_operator_hessian_diag_part_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        Hess = nnu.linalg.LinearOperatorHessian(func_1, x0)

        self.assertAllClose(Hess.diag_part(), tf.linalg.diag_part(Hess.to_dense()))

    def test_linalg_linear_operator_hessian_trace_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)

        Hess = nnu.linalg.LinearOperatorHessian(func_1, x0)

        self.assertAllClose(Hess.trace(), tf.linalg.trace(Hess.to_dense()))

    def test_linalg_linear_operator_hessian_quadratic_form_1(self):
        shape = (16, 5)

        x0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v0 = tf.random.normal(shape=shape, dtype=tf.float32)
        v1 = tf.random.normal(shape=shape, dtype=tf.float32)

        Hess = nnu.linalg.LinearOperatorHessian(func_1, x0)

        v0Hv1 = tf.linalg.matvec(tf.expand_dims(v0, axis=-2), tf.linalg.matvec(Hess.to_dense(), v1))
        v0Hv1 = tf.reshape(v0Hv1, shape=shape[:-1])

        self.assertAllClose(Hess.quadratic_form(v0, v1), v0Hv1)



if __name__ == '__main__':
    tf.test.main()
