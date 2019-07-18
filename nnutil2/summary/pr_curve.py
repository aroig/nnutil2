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

def pr_curve(name, predictions, truth, streaming=False):
    num_threshold = 200

    if streaming:
        summary, update_op = tb.summary.pr_curve_streaming_op(
            name,
            predictions=predictions,
            labels=tf.cast(truth, tf.bool),
            num_thresholds=num_threshold,
            metrics_collections=[tf.GraphKeys.SUMMARIES],
            updates_collections=[tf.GraphKeys.UPDATE_OPS])
    else:
        summary = tb.summary.pr_curve_streaming_op(
            name,
            predictions=predictions,
            labels=tf.cast(truth, tf.bool),
            num_thresholds=num_threshold)

    return summary

def label_pr_curve(y_pred, y_true, name=None, labels=None, streaming=False):
    if name is None:
        name = "pr_curve"

    if labels is None:
        labels = ['{}'.format(i) for i in range(0, labels.shape[-1])]

    summary_list = []
    with tf.name_scope(name):
        for i, lb in enumerate(labels):
            summary = pr_curve(lb, predictions[:, i], tf.equal(labels, i), streaming=streaming)
            summary_list.append(summary)

        merged_summary = tf.summary.merge(summary_list)

    return merged_summary
