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


import tensorflow as tf

class LinearOperatorHessian(tf.linalg.LinearOperator):
    def __init__(self, func, x,
                 is_non_singular=None, is_positive_definite=None,
                 use_pfor = True,
                 name="LinearOperatorHessian"):
        self._func = func
        self._x = x

        self._batch_shape = x.shape[:-1]
        self._batch_size = self._batch_shape.num_elements()
        self._inner_size = x.shape[-1]

        self._use_pfor = use_pfor

        super(LinearOperatorHessian, self).__init__(
            dtype=self._x.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=True,
            is_positive_definite=is_positive_definite,
            is_square=True,
            name=name)

    def _shape(self):
        return self._batch_shape + (self._inner_size, self._inner_size)

    def _assert_self_adjoint(self):
        return tf.no_op("assert_self_adjoint")

    def _directional_derivative(self, func, x, v, adjoint_arg=False):
        assert v.shape.rank == self._batch_shape.rank + 2
        assert v.shape[:-2] == self._batch_shape

        assert x.shape == self._batch_shape + (self._inner_size,)

        if adjoint_arg:
            num_vectors = v.shape[-2]
            inner_size = v.shape[-1]

        else:
            num_vectors = v.shape[-1]
            inner_size = v.shape[-2]

        assert v.dtype == x.dtype
        assert inner_size == self._inner_size

        with tf.GradientTape() as tape:
            s = tf.constant(0, shape=(num_vectors,), dtype=x.dtype)
            tape.watch(s)

            y = func(x + tf.linalg.matvec(v, s, adjoint_a=adjoint_arg))

        y_v = tape.jacobian(y, s, experimental_use_pfor=self._use_pfor)
        assert y_v.shape == self._batch_shape + (num_vectors,)

        return y_v

    def _flat_jacobian(self, x_semi_flat):
        with tf.GradientTape() as tape:
            tape.watch(x_semi_flat)

            x_unflat = tf.reshape(x_semi_flat, shape=self._x.shape)

            y = self._func(x_unflat)
            assert y.shape == self._batch_shape

            y_semi_flat = tf.reshape(y, shape=(self._batch_size,1))

        y_x_semi_flat = tape.batch_jacobian(y_semi_flat, x_semi_flat, experimental_use_pfor=self._use_pfor)
        assert y_x_semi_flat.shape == tf.TensorShape([self._batch_size, 1, self._inner_size])

        y_x_semi_flat = tf.reshape(y_x_semi_flat, shape=(self._batch_size, self._inner_size))
        return y_x_semi_flat

    def _matmul(self, v, adjoint=False, adjoint_arg=False):
        assert v.shape.rank == self._batch_shape.rank + 2

        x_flat = tf.reshape(self._x, shape=(self._batch_size, self._inner_size))
        v_flat = tf.reshape(v, shape=(self._batch_size,) + v.shape[-2:])

        with tf.GradientTape() as tape:
            tape.watch(x_flat)

            x_unflat = tf.reshape(x_flat, shape=self._x.shape)
            v_unflat = tf.reshape(v_flat, shape=v.shape)

            y_v = self._directional_derivative(self._func, x_unflat, v_unflat, adjoint_arg=False)

            num_vectors = y_v.shape[-1]
            y_v_flat = tf.reshape(y_v, shape=(self._batch_size, num_vectors))

        Hv_flat = tape.batch_jacobian(y_v_flat, x_flat, experimental_use_pfor=self._use_pfor)
        assert Hv_flat.shape == tf.TensorShape([self._batch_size, num_vectors, self._inner_size])

        Hv_flat_t = tf.transpose(Hv_flat, perm=[0, 2, 1])
        Hv = tf.reshape(Hv_flat_t, shape=self._batch_shape + (self._inner_size, num_vectors))

        return Hv

    def _diag_part(self):
        x_flat = tf.reshape(self._x, shape=(self._batch_size * self._inner_size, 1))

        with tf.GradientTape() as tape:
            tape.watch(x_flat)

            x_semi_flat = tf.reshape(x_flat, shape=(self._batch_size, self._inner_size))

            y_x_semi_flat = self._flat_jacobian(x_semi_flat)
            assert y_x_semi_flat.shape == tf.TensorShape([self._batch_size, self._inner_size])

            y_x_flat = tf.reshape(y_x_semi_flat, shape=(self._batch_size * self._inner_size, 1))

        y_xx_flat = tape.batch_jacobian(y_x_flat, x_flat, experimental_use_pfor=self._use_pfor)
        assert y_xx_flat.shape == tf.TensorShape([self._batch_size * self._inner_size, 1, 1])

        y_xx = tf.reshape(y_xx_flat, shape=self._batch_shape + (self._inner_size,))

        return y_xx

    def _trace(self):
        H_diag = self._diag_part()
        tr = tf.reduce_sum(H_diag, axis=-1)
        return tr

    def _to_dense(self):
        x_flat = tf.reshape(self._x, shape=(self._batch_size, self._inner_size))

        with tf.GradientTape() as tape:
            tape.watch(x_flat)

            y_x_flat = self._flat_jacobian(x_flat)
            assert y_x_flat.shape == tf.TensorShape([self._batch_size, self._inner_size])

        y_xx_flat = tape.batch_jacobian(y_x_flat, x_flat, experimental_use_pfor=self._use_pfor)
        assert y_xx_flat.shape == tf.TensorShape([self._batch_size, self._inner_size, self._inner_size])

        y_xx = tf.reshape(y_xx_flat, shape=self._batch_shape + (self._inner_size, self._inner_size))
        return y_xx

    def directional_derivative(self, v, name="directional_derivative"):
        assert v.shape == self._x.shape

        with self._name_scope(name):
            v = tf.expand_dims(v, axis=-1)
            y_v = self._directional_derivative(self._func, self._x, v)
            assert y_v.shape == self._batch_shape + (1,)

            y_v = tf.reshape(y_v, shape=self._batch_shape)
            return y_v

    def quadratic_form(self, v0, v1, name="quadratic_form"):
        assert v0.shape == self._x.shape
        assert v1.shape == self._x.shape

        with self._name_scope(name):
            v0 = tf.expand_dims(v0, axis=-1)
            v1 = tf.expand_dims(v1, axis=-1)

            def first_derivative(x):
                y_v0 = self._directional_derivative(self._func, x, v0)
                assert y_v0.shape == self._batch_shape + (1,)
                y_v0 = tf.reshape(y_v0, shape=self._batch_shape)
                return y_v0

            y_v0v1 = self._directional_derivative(first_derivative, self._x, v1)
            assert y_v0v1.shape == self._batch_shape + (1,)

            y_v0v1 = tf.reshape(y_v0v1, shape=self._batch_shape)
            return y_v0v1
