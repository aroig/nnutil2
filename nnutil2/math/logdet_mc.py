#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2019, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

from .trace_mc import trace_mc


def logdet_mc(f, shape=None, batch_rank: int = 1, seed=None):
    """Computes an unbiased Monte-Carlo approximation for log det A

       A is defined implicitly, f(v) = A v

       log det (1 + A) = tr log (1 + A) = A - A^2/2 + A^3/3 + ...
    """
    assert shape is not None

    def A(v):
        return f(v) - v

    trA = trace_mc(A, shape=shape, batch_rank=batch_rank, seed=seed)

    logdet = trA
    return logdet
