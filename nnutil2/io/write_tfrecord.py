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

import tensorflow as tf

import nnutil2 as nnu

def write_tfrecord(dataset, path):
    """ Writes a dataset into a tfrecord file
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    writer = tf.io.TFRecordWriter(path)
    for x in dataset:
        features = tf.train.Features(feature=nnu.nest.as_feature(x))
        example = tf.train.Example(features=features)
        example = example.SerializeToString()
        writer.write(example)
