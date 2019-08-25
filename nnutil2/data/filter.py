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

class Filter(tf.data.Dataset):
    """Filter structured dataset by a set of keys"""
    def __init__(self, dataset, keys=None):
        assert keys is not None
        self._keys = set(keys)

        self._input_datasets = [dataset]

        dataset = dataset.map(self.filter_structure)
        self._dataset = dataset

        super().__init__(self._dataset._variant_tensor)

    def filter_structure(self, features):
        features_filt = {k: v for k, v in features.items() if k in self._keys}
        return features_filt

    def _inputs(self):
        return list(self._input_datasets)

    @property
    def element_spec(self):
        return self._dataset.element_spec
