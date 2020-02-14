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


import tensorflow as tf
import numpy as np

class RandomNormal(tf.data.Dataset):
    def __init__(self, shape, dtype=tf.float32, seed=None):
        self._shape = shape

        dataset = tf.data.Dataset.from_tensors(tf.random.normal(shape=shape, dtype=dtype, seed=seed))
        dataset = dataset.repeat()
        self._dataset = dataset

        super().__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
