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

class LinearOperatorAntisymmetrize(tf.linalg.LinearOperator):
    def __init__(self,
                 func,
                 is_non_singular=None,
                 name="LinearOperatorAntisymmetrize"):
        self._func = func

        assert self._func.is_square

        super(LinearOperatorAntisymmetrize, self).__init__(
            dtype=self._func.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=False,
            is_positive_definite=self._func.is_positive_definite,
            is_square=True,
            name=name)

    def _shape(self):
        return self._func.shape

    def _matmul(self, v, adjoint=False, adjoint_arg=False):
        Av = self._func.matmul(v, adjoint=False, adjoint_arg=adjoint_arg)
        Atv = self._func.matmul(v, adjoint=True, adjoint_arg=adjoint_arg)
        return 0.5 * (Av - Atv)

    def _diag_part(self):
        zero = tf.zeros(dtype=self._func.dtype, shape=self._func.shape[:-1])
        return zero

    def _to_dense(self):
        A = self._func.to_dense()
        return 0.5 * (A - tf.linalg.adjoint(A))
