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

from .dotprod import dotprod
from .orthogonalize import orthogonalize

def symmetric_lanczos(f, vinit, size: int, v0=None, orthogonalize_step: bool = True) -> tf.Tensor:
    """Lanczos algorithm https://en.wikipedia.org/wiki/Lanczos_algorithm.

       Given A is a symmetric tensor of shape (..., N, N), the lanczos
       algorithm computes a low rank approximation of it that aims to
       approximate the extremal eigenvalues.

       f(v) is a function that computes A v implicitly.
       vinit() is a function that produces tensor of shape (..., N) to seed the Lanczos iteration.

       returns tensors (T, V) such that:

       1. V is orthogonal of shape (..., N, size) and spans the Krillov
          space {v0, f(v0), f^2(v0) ..., f^size(v0)}

       2. T = V* A V, is symmetric, tridiagonal of shape (..., size, size)

    """
    assert isinstance(f, tf.linalg.LinearOperator)

    if v0 is None:
        v0 = vinit()

    v0, _ = tf.linalg.normalize(v0, axis=-1)
    dtype = v0.dtype

    v_shape = v0.shape
    N = v_shape[-1]
    batch_shape = v_shape[:-1]

    assert size >= 2
    alpha_shape = batch_shape + (size,)
    beta_shape = batch_shape + (size,)
    T_shape = batch_shape + (size, size)
    V_shape = batch_shape + (N, size)

    fv0 = f.matvec(v0)
    assert fv0.shape == v0.shape

    alpha0 = dotprod(fv0, v0, keepdims=True)
    assert alpha0.shape == batch_shape + (1,)

    w0 = fv0 - alpha0 * v0
    beta0 = tf.zeros(shape=batch_shape + (1,), dtype=dtype)
    V0 = tf.expand_dims(v0, axis=-1)

    i0 = tf.constant(0, dtype=tf.int32)

    state0 = (
        i0,
        alpha0,
        beta0,
        V0,
        w0,
        v0,
    )

    shape_invariants = (
        i0.shape,
        batch_shape + tf.TensorShape([None]),
        batch_shape + tf.TensorShape([None]),
        batch_shape + tf.TensorShape([N, None]),
        batch_shape + tf.TensorShape([N]),
        batch_shape + tf.TensorShape([N]),
    )

    def lanczos_cond(i, alpha, beta, V, w, v):
        return i < size-1

    def lanczos_body(i, alpha, beta, V, w, v):
        v_ip1, beta_ip1 = tf.linalg.normalize(w, axis=-1)

        # TODO: handle case where beta_new is < epsilon by creating a new initial condition

        if orthogonalize_step:
            v_ip1 = orthogonalize(v_ip1, V)

        assert v.shape == v_ip1.shape

        # Lanczos update
        fv_ip1 = f.matvec(v_ip1)
        alpha_ip1 = dotprod(fv_ip1, v_ip1, keepdims=True)
        w_ip1 = fv_ip1 - alpha_ip1 * v_ip1 - beta_ip1 * v
        assert w_ip1.shape == w.shape

        # Accumulate outputs
        alpha = tf.concat([alpha, alpha_ip1], axis=-1)
        beta = tf.concat([beta, beta_ip1], axis=-1)
        V = tf.concat([V, tf.expand_dims(v_ip1, axis=-1)], axis=-1)

        return (i+1, alpha, beta, V, w_ip1, v_ip1)

    (N, alpha, beta, V, w, v) = tf.while_loop(
        lanczos_cond,
        lanczos_body,
        state0,
        shape_invariants=shape_invariants
    )

    alpha = tf.reshape(alpha, shape=alpha_shape)
    beta = tf.reshape(beta, shape=beta_shape)
    V = tf.reshape(V, shape=V_shape)

    T_alpha = tf.linalg.diag(alpha)
    T_beta = tf.roll(tf.linalg.diag(beta), shift=-1, axis=-1)
    T = T_alpha + T_beta + tf.linalg.matrix_transpose(T_beta)

    assert V.shape == V_shape
    assert T.shape == T_shape

    return (T, V)
