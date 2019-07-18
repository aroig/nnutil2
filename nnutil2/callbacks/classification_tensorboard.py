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

from .tensorboard import TensorBoard

import nnutil2 as nnu


class ClassificationTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super(ClassificationTensorBoard, self).__init__(**kwargs)

    def _common_summaries(self, writer, logs, mode_prefix, prefix, step):
        super(ClassificationTensorBoard, self)._common_summaries(writer, logs, mode_prefix, prefix, step)

        # scalars
        for name in ['accuracy', 'cross_entropy']:
            name = mode_prefix + name
            if name in logs:
                tf.summary.scalar(prefix + name, logs[name], step=step)

        # Confusion matrix
        for name in ['confusion_matrix']:
            name = mode_prefix + name
            if name in logs:
                nnu.summary.confusion_matrix(
                    logs[name],
                    step=step,
                    name=prefix + name,
                    labels=self.model.labels)

        # TODO: need to get per-epoch y_pred and y_true
        # for i, lb in enumerate(self._labels):
        #     nnu.summary.pr_curve(lb, y_pred[:, i], tf.equal(y_true, i))

        # TODO: save hyperparameters

    def _train_summaries(self, writer, logs, prefix, step):
        self._common_summaries(writer, logs, "", prefix, step)

    def _eval_summaries(self, writer, logs, prefix, step):
        self._common_summaries(writer, logs, "val_", prefix, step)
