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

from ..random import uniform_unit_vector
from .symmetric_lanczos import symmetric_lanczos


def generalized_quadratic_form(A: tf.Tensor, g, u: tf.Tensor, v: tf.Tensor):
    """Evaluates a quadratic form on g(A).

    Let λ_i, w_i an eigenvalue decomposition for A.

    A = W Λ W^t

    <u, g(A) v> = sum_i <u, w_i> g(λ_i) <w_i, v>

    The matrix A  is a tensor of shape (..., N, N)
    The function g takes a tensor of any shape and applies an entry-wise scalar function.
    The vectors u, v are tensors of shape (..., N)

    """

    theta, W = tf.linalg.eigh(A)
    gtheta = g(theta)
    assert theta.shape == gtheta.shape

    Wtu = tf.linalg.matvec(W, u, transpose_a=True)
    Wtv = tf.linalg.matvec(W, v, transpose_a=True)

    value = tf.reduce_sum(Wtu * gtheta * Wtv, axis=-1)
    return value


def generalized_quadratic_form_mc(fA, g, u: tf.Tensor, v: tf.Tensor, shape,
                                  lanczos_size: int = 4, num_samples: int = 1):
    """Monte-Carlo approximation of the generalized quadratic form evaluation of a symmetric matrix.

    <u, g(A) v> = sum_i <u, w_i> g(λ_i) <w_i, v>

    The matrix A is given implicitly through a function f(v) = A v where v is a tensor of shape (..., N)
    The function g takes a tensor of any shape and applies an entry-wise scalar function.
    The vectors u, v are tensors of shape (..., N)

    Let T, U be a lanczos approximation for v.
    Let λ_i, w_i be the eigenvalue decomposition of T.

    <u, g(A) v> = sum_i <U u, w_i> g(λ_i) <w_i, v>
                = sum_i <U u, w_i> g(λ_i)

    """

    N = shape[-1]

    def trace_single_sample(dummy):
        T, V = symmetric_lanczos(
            fA,
            lambda : uniform_unit_vector(shape=shape),
            size=lanczos_size,
            orthogonalize_step=True
        )

        # Compute generalized trace via Gauss quadrature
        theta, eigenvec = tf.linalg.eigh(T)
        gtheta = g(theta)
        assert theta.shape == gtheta.shape

        tau = eigenvec[..., 0, :]
        tr = N * tf.reduce_sum(gtheta * tau * tau, axis=-1)

        return tr

    tr_samples = tf.map_fn(trace_single_sample, tf.zeros(shape=(num_samples,)))
    tr = tf.reduce_mean(tr_samples, axis=0)
    assert tr.shape == shape[:-1]

    return tr
