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

def pr_curve(data, name="pr_curve", step=None, description=None):
    data = tf.cast(data, dtype=tf.float32)

    num_thresholds = data.shape[-1]
    tf.debugging.assert_shapes([
        (data, (2, 2, num_thresholds))
    ])

    summary_metadata = tb.plugins.pr_curve.metadata.create_summary_metadata(
        display_name=name,
        description=description,
        num_thresholds=num_thresholds
    )

    with tf.summary.experimental.summary_scope(name, 'pr_curve', values=[data, step]) as (tag, _):
        true_pos = tf.cast(data[1, 1, :], dtype=tf.float32)
        false_pos = tf.cast(data[0, 1, :], dtype=tf.float32)

        true_neg = tf.cast(data[0, 0, :], dtype=tf.float32)
        false_neg = tf.cast(data[1, 0, :], dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()

        precision = tf.math.divide_no_nan(true_pos + epsilon, true_pos + false_pos + epsilon)
        recall = tf.math.divide_no_nan(true_pos + epsilon, true_pos + false_neg + epsilon)

        # TODO: once tensorboard moves pr_curve to TF 2.0, we should use that instead.

        # Store values within a tensor. We store them in the order:
        # true positives, false positives, true negatives, false
        # negatives, precision, and recall.
        combined_data = tf.stack([
            tf.cast(true_pos, tf.float32),
            tf.cast(false_pos, tf.float32),
            tf.cast(true_neg, tf.float32),
            tf.cast(false_neg, tf.float32),
            tf.cast(precision, tf.float32),
            tf.cast(recall, tf.float32)
        ], axis=0)

        return tf.summary.write(
            tag=tag,
            tensor=combined_data,
            step=step,
            metadata=summary_metadata)
