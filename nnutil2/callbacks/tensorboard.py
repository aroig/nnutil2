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

class TensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoard, self).__init__(**kwargs)

    def _log_metrics(self, logs, prefix, step):
        """Writes metrics out as custom scalar summaries.

        Arguments:
            logs: Dict. Keys are scalar summary names, values are NumPy scalars.
            prefix: String. The prefix to apply to the scalar summary names.
            step: Int. The global step to use for TensorBoard.
        """
        if logs is None:
            logs = {}

        with tf.summary.record_if(True):
            train_writer = self._get_writer(self._train_run_name)
            with train_writer.as_default():
                self._train_summaries(train_writer, logs, prefix=prefix, step=step)

            eval_writer = self._get_writer(self._validation_run_name)
            with eval_writer.as_default():
                self._eval_summaries(eval_writer, logs, prefix=prefix, step=step)

    def _common_summaries(self, writer, logs, mode_prefix, prefix, step):
        # scalars
        for name in ['loss']:
            name = mode_prefix + name
            if name in logs:
                tf.summary.scalar(prefix + name, logs[name], step=step)

    def _train_summaries(self, writer, logs, prefix, step):
        self._common_summaries(writer, logs, "", prefix, step)

    def _eval_summaries(self, writer, logs, prefix, step):
        self._common_summaries(writer, logs, "val_", prefix, step)
