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

import itertools
import math

import tensorflow as tf

from ..util import normalize_axis

def permutation_sign(perm):
    if len(perm) == 1:
        return 1

    trans = 0

    for i in range(0,len(perm)):
        j = i + 1

        for j in range(j, len(perm)):
            if perm[i] > perm[j]:
                trans += 1

    if (trans % 2) == 0:
        return 1
    else:
        return -1

def symmetrize(x, axis):
    assert x.shape.rank >= len(axis)

    dtype = x.dtype
    shape = x.shape
    axis = normalize_axis(shape, axis)

    for a in axis:
        assert x.shape[axis[0]] == x.shape[a]

    y = tf.zeros(shape=shape, dtype=dtype)

    for axis_perm in itertools.permutations(range(0, len(axis))):
        perm = list(range(0, shape.rank))
        for a, pi in zip(axis, axis_perm):
            perm[a] = axis[pi]

        y += tf.transpose(x, perm)

    return y / math.factorial(len(axis))


def antisymmetrize(x, axis):
    assert x.shape.rank >= len(axis)

    dtype = x.dtype
    shape = x.shape
    axis = normalize_axis(shape, axis)

    for a in axis:
        assert x.shape[axis[0]] == x.shape[a]

    y = tf.zeros(shape=shape, dtype=dtype)

    for axis_perm in itertools.permutations(range(0, len(axis))):
        perm = list(range(0, shape.rank))
        for a, pi in zip(axis, axis_perm):
            perm[a] = axis[pi]

        y += permutation_sign(perm) * tf.transpose(x, perm)

    return y / math.factorial(len(axis))
