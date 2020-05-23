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

class LayersReplicaDense(tf.test.TestCase):
    def test_layer_replica_dense_1(self):
        x = tf.random.normal(shape=(2, 3, 5), dtype=tf.float32)
        lay = nnu.layers.ReplicaDense(nfilters=[7], replica_axes=[0], contraction_axes=[1])
        y = lay(x)

        self.assertAllEqual(tf.TensorShape([2, 7, 5]), y.shape)

if __name__ == '__main__':
    tf.test.main()
