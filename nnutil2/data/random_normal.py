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
    def __init__(self, tensor_spec, dtype=tf.float32, seed=None):
        self._tensor_spec = tensor_spec

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

        flat_spec = tf.nest.flatten(self._tensor_spec)
        size = len(flat_spec)

        random_normals = []
        for i, spec in enumerate(flat_spec):
            seed0 = size * value + i
            seed = tf.stack([seed0, seed0 + self._seed])
            shape = spec.shape
            dtype = spec.dtype
            dist = tf.random.stateless_normal(shape=shape, dtype=dtype, seed=seed)
            random_normals.append(dist)

        tensor = tf.nest.pack_sequence_as(self._tensor_spec, random_normals)
        return tensor

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
