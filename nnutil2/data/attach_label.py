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

class AttachLabel(tf.data.Dataset):
    def __init__(self, dataset, label_feature='label', onehot=False, labels=None):
        self._labels = labels
        self._onehot = onehot
        self._label_feature = label_feature
        self._input_datasets = [dataset]

        dataset = dataset.map(self.attach_label)
        self._dataset = dataset

        super().__init__(self._dataset._variant_tensor)

    def attach_label(self, feature):
        label = feature[self._label_feature]

        if self._labels is not None and label.dtype in set([tf.string]):
            label = tf.py_function(self.label_index_fn, [label], tf.int32)
            label = tf.reshape(label, shape=())

        if self._onehot:
            label = tf.one_hot(label, len(self._labels))

        return (feature, label)

    def label_index_fn(self, label):
        try:
            idx = self._labels.index(label.numpy().decode())
        except ValueError:
            idx = -1

        return idx

    def _inputs(self):
        return list(self._input_datasets)

    @property
    def element_spec(self):
        return self._dataset.element_spec
