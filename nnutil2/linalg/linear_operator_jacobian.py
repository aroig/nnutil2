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

from ..util import as_shape, batch_shape

class LinearOperatorJacobian(tf.linalg.LinearOperator):
    def __init__(self,
                 func,
                 x,
                 input_shape=None,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 use_pfor=True,
                 name="LinearOperatorJacobian"):
        self._func = func
        self._x = x

        if input_shape is None:
            input_shape = x.shape

        self._inner_in_shape = as_shape(input_shape)
        self._inner_in_size = self._inner_in_shape.num_elements()

        self._batch_shape = batch_shape(x, self._inner_in_shape)
        self._batch_size = self._batch_shape.num_elements()

        y = self._func(x)
        self._inner_out_shape = as_shape(y.shape[self._batch_shape.rank:])
        self._inner_out_size = self._inner_out_shape.num_elements()

        self._use_pfor = use_pfor

        super(LinearOperatorJacobian, self).__init__(
            dtype=self._x.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=(self._inner_in_shape == self._inner_out_shape),
            name=name)

    def _shape(self):
        return self._batch_shape + (self._inner_out_size, self._inner_in_size)

    def _flat_evaluation(self, x_flat):
        x = tf.reshape(x_flat, shape=self._batch_shape + self._inner_in_shape)
        y = self._func(x)
        y_flat = tf.reshape(y, shape=(self._batch_size, self._inner_out_size))
        return y_flat

    def _matmul(self, v, adjoint=False, adjoint_arg=False):
        assert v.shape.rank == self._batch_shape.rank + 2

        x_flat = tf.reshape(self._x, shape=(self._batch_size, self._inner_in_size))
        v_flat = tf.reshape(v, shape=(self._batch_size,) + v.shape[-2:])

        if adjoint_arg:
            num_vectors = v.shape[-2]
        else:
            num_vectors = v.shape[-1]

        if adjoint:
            with tf.GradientTape() as tape:
                tape.watch(x_flat)
                tape.watch(v_flat)
                y_flat = self._flat_evaluation(x_flat)
                y_flat = tf.linalg.matvec(v_flat, y_flat, adjoint_a=not adjoint_arg)

            y_x_flat = tape.batch_jacobian(y_flat, x_flat, experimental_use_pfor=self._use_pfor)
            res = tf.linalg.matrix_transpose(y_x_flat)

        else:
            with tf.GradientTape() as tape:
                s = tf.zeros(shape=(self._batch_size, num_vectors), dtype=v.dtype)
                tape.watch(x_flat)
                tape.watch(s)

                vs_flat = tf.linalg.matvec(v_flat, s, adjoint_a=adjoint_arg)
                y_flat = self._flat_evaluation(x_flat + vs_flat)

            res = tape.batch_jacobian(y_flat, s, experimental_use_pfor=self._use_pfor)

        return res

    def _diag_part(self):
        size = min(self._inner_in_size, self._inner_out_size)

        x_flat = tf.reshape(self._x, shape=(self._batch_size, self._inner_in_size))
        x_flat = tf.reshape(x_flat[...,:size], shape=(self._batch_size * size, 1))

        with tf.GradientTape() as tape:
            tape.watch(x_flat)

            x_unflat = tf.reshape(x_flat, shape=(self._batch_size, size))
            y_unflat = self._flat_evaluation(x_unflat)
            y_flat = tf.reshape(y_unflat[..., :size], shape=(self._batch_size * size, 1))

        diag_flat = tape.batch_jacobian(y_flat, x_flat, experimental_use_pfor=self._use_pfor)
        diag = tf.reshape(diag_flat, shape=(self._batch_size, size))
        return diag

    def _to_dense(self):
        x_flat = tf.reshape(self._x, shape=(self._batch_size, self._inner_in_size))

        with tf.GradientTape() as tape:
            tape.watch(x_flat)

            y_flat = self._flat_evaluation(x_flat)
            assert y_flat.shape == tf.TensorShape([self._batch_size, self._inner_out_size])

        y_x_flat = tape.batch_jacobian(y_flat, x_flat, experimental_use_pfor=self._use_pfor)
        assert y_x_flat.shape == tf.TensorShape([self._batch_size, self._inner_out_size, self._inner_in_size])

        y_x = tf.reshape(y_x_flat, shape=self._batch_shape + (self._inner_out_size, self._inner_in_size))
        return y_x
