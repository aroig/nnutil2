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
from .generalized_matvec import generalized_approximate_matvec

def generalized_killing_form(A, B, g=None, h=None):
    """Generalized killing form of two symmetric matrices

    Let λ_i, v_i be an eigen decomposition for A
    Let μ_j, w_j be an eigen decomposition for B

    Kill(A, B) = tr g(A) h(B)
               = sum_i <v_i, g(A)h(B) v_i>
               = sum_i, j g(λ_i) h(μ_j) <v_i, w_j>^2

    The matrices A, B are tensors of shape (..., N, N)
    The functions g, h take a tensor of any shape and apply an entry-wise scalar function.

    """

    theta0, W0 = tf.linalg.eigh(A)
    if g is not None:
        gtheta0 = g(theta0)
    assert theta0.shape == gtheta0.shape

    theta1, W1 = tf.linalg.eigh(A)
    if h is not None:
        htheta1 = h(theta1)

    assert theta1.shape == htheta1.shape

    gramm = tf.matmul(W0, W1, transpose_a=True)
    value = tf.reduce_sum(gtheta0 * tf.matvec(gramm, htheta1), axis=-1)
    return value


def generalized_killing_form_mc(fA, fB, shape, g=None, h=None,
                                lanczos_size: int = 4, num_samples: int = 1):
    """Monte-Carlo approximation of the generalized killing form of a symmetric matrix

    Let T0, V0 be a Lanczos approximation of A.
    Let T1, V1 be a Lanczos approximation of B.

    Let λ_i, w_i be an eigen decomposition of T0
    Let μ_j, s_j be an eigen decomposition of T1

    Kill(A, B) = tr g(A) h(B)
               = E_v <g(A) v, h(B) v>
               ≈ E_v <V0 g(T0) e_1, V1 h(T1) e_1>

    The matrix A is given implicitly through a function fA(v) = A v where v is a tensor of shape (..., N)
    The matrix B is given implicitly through a function fB(v) = B v where v is a tensor of shape (..., N)
    The functions g, h takes a tensor of any shape and applies an entry-wise scalar function.
    """

    N = shape[-1]

    def trace_single_sample(dummy):
        u = uniform_unit_vector(shape=shape)
        vleft = generalized_approximate_matvec(fA, u, g, lanczos_size=lanczos_size)
        vright = generalized_approximate_matvec(fB, u, h, lanczos_size=lanczos_size)

        tr = N * tf.reduce_sum(vleft * vright, axis=-1)
        return tr

    tr_samples = tf.map_fn(trace_single_sample, tf.zeros(shape=(num_samples,)))
    tr = tf.reduce_mean(tr_samples, axis=0)
    assert tr.shape == shape[:-1]

    return tr
