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
from .. import util

class RandomNormal(tf.data.Dataset):
    def __init__(self, shape, mean=0, stddev=1, dtype=tf.float32, seed=None):
        self._shape = util.as_shape(shape)
        self._dtype = dtype

        self._mean = mean
        self._stddev = stddev

        if seed is None:
            seed = 1
        self._seed = seed

        dataset = tf.data.experimental.Counter(
            start=0, step=1, dtype=tf.dtypes.int64
        )
        dataset = dataset.map(self.attach_random_normal)

        self._dataset = dataset

        super().__init__(self._dataset._variant_tensor)

    def attach_random_normal(self, value):

        flat_shape = tf.nest.flatten(self._shape)
        size = len(flat_shape)

        random_normals = []
        for i, shape in enumerate(flat_shape):
            seed0 = size * value + i
            seed = tf.stack([seed0, seed0 + self._seed])

            dist = tf.random.stateless_normal(
                shape=shape,
                mean=self._mean,
                stddev=self._stddev,
                dtype=self._dtype,
                seed=seed
            )

            random_normals.append(dist)

        tensor = tf.nest.pack_sequence_as(self._shape, random_normals)
        return tensor

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
