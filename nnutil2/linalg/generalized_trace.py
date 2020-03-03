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

def generalized_trace(A, g):
    """Generalized trace of a symmetric matrx

    tr g(A) = sum_i g(λ_i)

    The matrix A is a tensor of shape (..., N, N)
    The function g takes a tensor of any shape and applies an entry-wise scalar function.

    """

    theta = tf.linalg.eigvalsh(A)
    gtheta = g(theta)
    assert theta.shape == gtheta.shape

    tr = tf.reduce_sum(gtheta, axis=-1)
    return tr


def generalized_trace_mc(fA, g, shape, lanczos_size: int = 4, num_samples: int = 1):
    """Monte-Carlo approximation of the generalized trace of a symmetric matrx

    The matrix A is given implicitly through a function f(v) = A v where v is a tensor of shape (..., N)
    The function g takes a tensor of any shape and applies an entry-wise scalar function.

    See https://doi.org/10.1137/16M1104974

    Let T, V be a Lanczos approximation for A.
    Let λ_i, w_i be an eigen decomposition of T

    A ≈ V T Vt

    tr g(A) = E_v <v, g(A) v>
            ≈ E_v <Vt v, T Vt v>
            = E_v sum_i g(λ_i) <w_i, Vt v>^2
            ≈ E_v sum_i g(λ_i) <w_i, e_1>^2

    The last approximation is exact under orthogonality, because v is the first of the lanczos vectors.
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
        theta, W = tf.linalg.eigh(T)
        gtheta = g(theta)
        assert theta.shape == gtheta.shape

        tau = W[..., 0, :]
        tr = N * tf.reduce_sum(gtheta * tau * tau, axis=-1)

        return tr

    tr_samples = tf.map_fn(trace_single_sample, tf.zeros(shape=(num_samples,)))
    tr = tf.reduce_mean(tr_samples, axis=0)
    assert tr.shape == shape[:-1]

    return tr
