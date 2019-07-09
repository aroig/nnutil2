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

from tensorflow.python.keras import testing_utils

class LayersIdentity(tf.test.TestCase):
    # def test_layers_identity_1(self):
    #     testing_utils.layer_test(
    #         nnu.layers.Identity,
    #         kwargs={},
    #         input_shape=(2, 3))

    def test_layer_identity_2(self):
        with self.cached_session() as sess:
            x = tf.random.normal(shape=(2, 3), dtype=tf.float32)
            lay = nnu.layers.Identity()
            y = lay(x)

            xval, yval = sess.run([x, y])
            self.assertAllEqual(xval, yval)


if __name__ == '__main__':
    tf.test.main()
