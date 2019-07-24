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


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", nlabels=None, dtype=tf.float32):
        super(F1Score, self).__init__(name=name, dtype=dtype)

        assert nlabels is not None

        self._nlabels = nlabels
        self._shape = (nlabels, nlabels)

        self._confusion_matrix = self.add_weight(
            'confusion_matrix',
            shape=self._shape,
            initializer=ks.initializers.Zeros
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_pred = tf.reshape(y_pred, shape=(-1,))
        y_true = tf.reshape(y_true, shape=(-1,))

        y_pred.shape.assert_is_compatible_with(y_true.shape)

        confusion = tf.math.confusion_matrix(y_true, y_pred)
        confusion = tf.cast(confusion, dtype=self._dtype)
        update_op = self._confusion_matrix.assign_add(confusion)

        return update_op

    def result(self):
        epsilon = tf.keras.backend.epsilon()
        precision = tf.math.divide_no_nan(tf.linalg.diag_part(self._confusion_matrix) + epsilon,
                                          tf.reduce_sum(self._confusion_matrix, axis=0) + epsilon)

        recall = tf.math.divide_no_nan(tf.linalg.diag_part(self._confusion_matrix) + epsilon,
                                          tf.reduce_sum(self._confusion_matrix, axis=1) + epsilon)

        per_class_f1 = tf.math.divide_no_nan(2 * precision * recall, (precision + recall))
        result = tf.reduce_mean(per_class_f1)

        return result

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros(shape=self._shape))

    def get_config(self):
        config = super(F1Score, self).get_config()
        config.update({
            'nlabels': self._nlabels
        })
        return config
