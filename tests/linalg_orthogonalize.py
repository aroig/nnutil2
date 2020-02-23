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

import unittest

import numpy as np
import tensorflow as tf

import nnutil2 as nnu

class LinalgOrthogonalize(tf.test.TestCase):
    def setUp(self):
        pass

    def test_linalg_orthogonalize_1(self):
        v = tf.constant([[1, 1, 1], [1, 1, 1]], dtype=tf.float32)
        A = tf.constant([[[1, 0, 0], [0, 1, 0]], [[1, -1, 0], [1, 1, 0]]], dtype=tf.float32)
        A = tf.linalg.matrix_transpose(A)
        v_ortho = tf.constant([[0, 0, 1], [0, 0, 1]], dtype=tf.float32)

        self.assertAllClose(v_ortho, nnu.linalg.orthogonalize(v, A))

if __name__ == '__main__':
    tf.test.main()
