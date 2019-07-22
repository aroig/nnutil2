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
import tensorflow.keras as ks

from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils


class PRCurve(tf.keras.metrics.Metric):
    def __init__(self, name="pr_curves", label=None, num_thresholds=64, dtype=tf.float32):
        super(PRCurve, self).__init__(name=name, dtype=dtype)

        assert label is not None

        self._label = label
        self._num_thresholds = num_thresholds

        self._thresholds = [float(i / (self._num_thresholds-1)) for i in range(0, self._num_thresholds)]

        self.true_positives = self.add_weight(
            'true_positives',
            shape=(self._num_thresholds,),
            initializer=ks.initializers.Zeros,
            dtype=self.dtype)

        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(self._num_thresholds,),
            initializer=ks.initializers.Zeros,
            dtype=self.dtype)

        self.false_positives = self.add_weight(
            'false_positives',
            shape=(self._num_thresholds,),
            initializer=ks.initializers.Zeros,
            dtype=self.dtype)

        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(self._num_thresholds,),
            initializer=ks.initializers.Zeros,
            dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[:, self._label]

        y_true = tf.equal(self._label, tf.cast(y_true, dtype=tf.int32))
        y_true = tf.reshape(y_true, shape=(-1,))

        update_op = metrics_utils.update_confusion_matrix_variables({
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
        }, y_true, y_pred, self._thresholds, sample_weight=sample_weight)

    def result(self):
        return tf.stack([
            tf.stack([self.true_negatives,  self.false_negatives], axis=0),
            tf.stack([self.false_positives, self.true_positives], axis=0)], axis=0)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros(shape=(self._num_thresholds, ), dtype=self.dtype))

    def get_config(self):
        config = super(PRCurve, self).get_config()
        config.extend({
            'label': self._label,
            'num_thresholds': self._num_thresholds
        })
        return config
