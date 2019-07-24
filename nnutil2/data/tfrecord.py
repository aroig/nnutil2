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

class TFRecord(tf.data.Dataset):
    def __init__(self, path=None, tensor_spec=None):
        assert path is not None
        assert tensor_spec is not None

        self._tensor_spec = tensor_spec

        if not os.path.exists(path):
            raise Exception("tfrecord file does not exist: {}".format(path))

        tfrecord_dataset = tf.data.TFRecordDataset(path)
        self._dataset = tfrecord_dataset.map(self.parse_example)

        super(TFRecord, self).__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return []

    @property
    def _element_structure(self):
        return tf.data.experimental.NestedStructure(self._tensor_spec)

    def parse_spec(self, tensor_spec):
        if type(tensor_spec) == dict:
            return {k: self.parse_spec(v) for k, v in tensor_spec.items()}

        elif type(tensor_spec) == tf.TensorSpec:
            return tf.io.FixedLenFeature(tensor_spec.shape, tensor_spec.dtype)

        elif type(tensor_spec) == tf.io.FixedLenFeature:
            return tensor_spec

        elif type(tensor_spec) == tf.io.VarLenFeature:
            return tensor_spec

        else:
            raise Exception("Unhandled input spec: {}".format(type(tensor_spec)))

    def parse_example(self, example_proto):
        parse_spec = self.parse_spec(self._tensor_spec)
        parsed_features = tf.io.parse_single_example(example_proto, parse_spec)

        return parsed_features
