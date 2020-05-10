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

import numpy as np
import tensorflow as tf
import nnutil2 as nnu

class DataRandomNormal(tf.test.TestCase):
    def test_flatten_vector(self):
        tf.random.set_seed(42)

        shape = { 'a': tf.TensorShape([2,3]), 'b': tf.TensorShape([]) }
        data = nnu.data.RandomNormal(shape=shape)

        for x in data.take(10):
            self.assertAllEqual(shape, nnu.util.as_shape(x))

if __name__ == '__main__':
    tf.test.main()
