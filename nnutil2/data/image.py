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

class Image(tf.data.Dataset):
    def __init__(self, dataset, key=None):
        self._input_datasets = [dataset]
        self._key = key

        dataset = dataset.map(self._process_image)

        self._dataset = dataset
        super().__init__(self._dataset._variant_tensor)

    def _process_image(self, x):
        image = x[self._key]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        x[self._key] = image
        return x

    def _inputs(self):
        return list(self._input_datasets)

    @property
    def element_spec(self):
        return self._dataset.element_spec
