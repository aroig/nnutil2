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
from tensorboard.plugins.hparams import api as hp

class TensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, run_id=None, **kwargs):
        self._run_id = run_id

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
            metric_name = mode_prefix + name
            if metric_name in logs:
                value = logs[metric_name]
                tf.summary.scalar(name, value, step=step)

    def _train_summaries(self, writer, logs, prefix, step):
        self._common_summaries(writer, logs, "", prefix, step)

    def _eval_summaries(self, writer, logs, prefix, step):
        self._common_summaries(writer, logs, "val_", prefix, step)

    def on_test_begin(self, logs=None):
        eval_writer = self._get_writer(self._validation_run_name)
        with eval_writer.as_default():
            hp.hparams_config(
                hparams=[hp.HParam(k) for k, v in self.model.hparams.items()],
                metrics=[hp.Metric("accuracy"), hp.Metric("f1_score")],
            )

        return super(TensorBoard, self).on_train_begin(logs=logs)

    def on_test_end(self, logs=None):
        eval_writer = self._get_writer(self._validation_run_name)
        with eval_writer.as_default():
            hp.hparams(hparams=self.model.hparams, trial_id=self._run_id)

        return super(TensorBoard, self).on_train_end(logs=logs)
