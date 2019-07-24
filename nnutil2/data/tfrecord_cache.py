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

from .tfrecord import TFRecord

class TFRecordCache(tf.data.Dataset):
    """ Cache dataset into a tfrecord file.

        If the tfrecord file exists, it does not consume the original dataset, otherwise it
        regenerates the tfrecord transparently.
    """

    def __init__(self, dataset, path):
        if self._needs_rebuild(path):
            print("Preparing tfrecord: {}".format(path))
            nnu.io.write_tfrecord(dataset, path)

        self._tensor_spec = nnu.nest.as_tensor_spec(dataset._element_structure)

        if not os.path.exists(path):
            raise Exception("tfrecord file does not exist: {}".format(path))

        self._dataset = TFRecord(path=path, tensor_spec=self._tensor_spec)

        super(TFRecordCache, self).__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return []

    @property
    def _element_structure(self):
        return tf.data.experimental.NestedStructure(self._tensor_spec)

    def _needs_rebuild(self, path):
        return not os.path.exists(path)
