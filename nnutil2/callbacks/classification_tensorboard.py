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
    def __init__(self, model=None, **kwargs):
        assert model is not None
        self._model = model

        super(ClassificationTensorBoard, self).__init__(**kwargs)

    def _common_summaries(self, writer, logs, mode_prefix, prefix, step):
        super(ClassificationTensorBoard, self)._common_summaries(writer, logs, mode_prefix, prefix, step)

        # Scalars
        for name in ['accuracy', 'cross_entropy', 'f1_score']:
            metric_name = mode_prefix + name
            if metric_name in logs:
                value = logs[metric_name]
                tf.summary.scalar(name, value, step=step)

        # Confusion matrix
        for name in ['confusion_matrix']:
            metric_name = mode_prefix + name
            if metric_name in logs:
                value = logs[metric_name]
                nnu.summary.confusion_matrix(
                    value,
                    step=step,
                    name=name,
                    labels=self.model.labels)

        # PR Curves
        for name in ['pr_curve']:
            for lb in self.model.labels:
                full_name = "{}/{}".format(name, lb)
                metric_name = mode_prefix + full_name

                if metric_name in logs:
                    value = logs[metric_name]
                    nnu.summary.pr_curve(
                        value,
                        step=step,
                        name=full_name)

    def _layer_summaries(self, writer, logs, mode_prefix, prefix, step):
        # Layer sizes
        layer_sizes = []

        for ly in self._model.layers[0].flat_layers:
            size = sum([w.shape.num_elements() for w in ly.variables])
            layer_sizes.append(size)

        layer_sizes = tf.constant(layer_sizes, dtype=tf.float32)
        nnu.summary.distribution(layer_sizes, name="layer/size", step=step)

    def _train_summaries(self, writer, logs, prefix, step):
        mode_prefix = ""
        self._common_summaries(writer, logs, mode_prefix, prefix, step)
        self._layer_summaries(writer, logs, mode_prefix, prefix, step)

    def _eval_summaries(self, writer, logs, prefix, step):
        mode_prefix = "val_"
        self._common_summaries(writer, logs, mode_prefix, prefix, step)
