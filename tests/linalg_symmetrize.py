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

import unittest

import tensorflow as tf

import nnutil2 as nnu

class LinalgSymmetrize(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_symmetrize_1(self):
        N = 32
        batch_size = 4

        shape = (batch_size, N, N)

        A = tf.random.normal(shape=shape)

        A_sym = nnu.linalg.symmetrize(A, axis=[-1, -2])
        self.assertAllClose(A_sym, tf.linalg.matrix_transpose(A_sym))

        A_sym2 = nnu.linalg.symmetrize(A_sym, axis=[-1, -2])
        self.assertAllClose(A_sym, A_sym2)



    def test_linalg_antisymmetrize_1(self):
        N = 32
        batch_size = 4

        shape = (batch_size, N, N)

        A = tf.random.normal(shape=shape)

        A_ant = nnu.linalg.antisymmetrize(A, axis=[-1, -2])
        self.assertAllClose(A_ant, -tf.linalg.matrix_transpose(A_ant))

        A_ant2 = nnu.linalg.antisymmetrize(A_ant, axis=[-1, -2])
        self.assertAllClose(A_ant, A_ant2)

if __name__ == '__main__':
    tf.test.main()
