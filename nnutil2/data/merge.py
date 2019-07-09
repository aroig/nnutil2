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

class Merge(tf.data.Dataset):
    def __init__(self, datasets):
        self._input_datasets = [ds for ds in datasets]

        dataset = tf.data.Dataset.zip(tuple(datasets))
        dataset = dataset.map(self._merge_dicts)
        self._dataset = dataset

        super(Merge, self).__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return list(self._input_datasets)

    @property
    def _element_structure(self):
        return self._dataset._element_structure

    def _merge_dicts(self, *features):
        res = {}
        for f in features:
            for k in f:
                res[k] = f[k]
        return res
