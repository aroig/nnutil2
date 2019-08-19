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


import os

import numpy as np
import tensorflow as tf

import nnutil2 as nnu

class TFRecord(tf.data.Dataset):
    def __init__(self, paths=None, tensor_spec=None):
        assert paths is not None
        assert tensor_spec is not None

        self._tensor_spec = tensor_spec

        path_list = nnu.io.list_file_paths(paths, "\\.tfrecord$")

        for p in path_list:
            if not os.path.exists(p):
                raise Exception("tfrecord file does not exist: {}".format(p))

        tfrecord_dataset = tf.data.TFRecordDataset(path_list)
        self._dataset = tfrecord_dataset.map(self._parse_example)

        super(TFRecord, self).__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return []

    @property
    def _element_structure(self):
        return self._dataset._element_structure

    def _parse_example(self, example_proto):
        feature_spec = nnu.nest.as_feature_spec(self._tensor_spec)
        parsed_features = tf.io.parse_single_example(example_proto, feature_spec)

        return parsed_features
