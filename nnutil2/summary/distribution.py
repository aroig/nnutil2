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
import tensorboard as tb
import numpy as np

from tensorboard.plugins.histogram.metadata import create_summary_metadata

def distribution(name, dist):
    dist = dist / tf.reduce_sum(dist)

    def entry(i, x):
        return tf.stack([
            tf.constant(i-0.5, shape=(), dtype=tf.float32),
            tf.constant(i+0.5, shape=(), dtype=tf.float32),
            tf.cast(x, dtype=tf.float32)
        ])

    dist_entries = tf.stack([entry(i, p) for i, p in enumerate(tf.unstack(dist))])

    metadata = create_summary_metadata(
        display_name=name,
        description=None)

    dist_summary = tf.summary.tensor_summary(
        name,
        dist_entries,
        summary_metadata=metadata)

    return dist_summary
