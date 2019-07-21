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
from tensorflow.python.ops.losses import util as tf_losses_utils

import nnutil2 as nnu


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, name="confusion_matrix", labels=None, dtype=tf.float32):
        super(ConfusionMatrix, self).__init__(name=name, dtype=dtype)

        assert labels is not None

        self._labels = labels
        self._shape = (len(labels), len(labels))

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
        total = tf.math.reduce_sum(self._confusion_matrix, axis=(0, 1))
        result = tf.math.divide_no_nan(self._confusion_matrix, total)

        return result

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros(shape=self._shape))

    def get_config(self):
        config = super(ConfusionMatrix, self).get_config()
        config.extend({
            'labels': self._labels
        })
        return config
