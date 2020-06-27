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

def generalized_matvec(A: tf.Tensor, v: tf.Tensor, g=None):
    """Generalized matvec evaluation

    Let λ_i, u_i be an eigen decomposition for A

    g(A) v = sum_i g(λ_i) <u_i, v> u_i

    The matrix A is a tensors of shape (..., N, N)
    The vector v is a tensor of shape (..., N)
    The function g takes a tensor of any shape and apply an entry-wise scalar function.

    """
    assert A.shape[:-1] == v.shape

    theta, U = tf.linalg.eigh(A)
    if g is not None:
        gtheta = g(theta)
    assert theta.shape == gtheta.shape

    Utv = tf.linalg.matvec(U, v, transpose_a=True)

    value = tf.linalg.matvec(U, Utv * gtheta)
    return value


def generalized_approximate_matvec(fA, u: tf.Tensor, g=None, lanczos_size: int = 4):
    """ Approximate generalized matvec

    The matrix A is given implicitly through a function fA(v) = A v where v is a tensor of shape (..., N)
    The vector u is a tensor of shape (..., N)
    The functions g, h takes a tensor of any shape and applies an entry-wise scalar function.

    Let T, V be a Lanczos approximation of A.
    Let λ_i, w_i an eigenvalue decomposition of T

    T = W Λ W^t
    T = V^t A V
    A ≈ V T V^t = V W Λ W^t V^t

    g(A) u ≈ V W g(Λ) W^t V^t u
           ≈ V W g(Λ) W^t e_1

    """

    if g is None:
        return fA(u)

    shape = u.shape

    def initial_state():
        yield u

        while True:
            yield uniform_unit_vector(shape=shape)

    T, V = symmetric_lanczos(
        fA,
        initial_state,
        size=lanczos_size,
        orthogonalize_step=True
    )

    theta, W = tf.linalg.eigh(T)
    gtheta = g(theta)
    assert theta.shape == gtheta.shape

    VW = V @ W
    VWtu = W[..., 0, :]
    value = tf.linalg.matvec(VW, gtheta * VWtu)
    return value
