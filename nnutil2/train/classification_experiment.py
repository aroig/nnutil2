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

from .experiment import Experiment

import nnutil2 as nnu

class ClassificationExperiment(Experiment):
    def __init__(self, labels=None, **kwargs):
        assert labels is not None
        self._labels = labels

        super(ClassificationExperiment, self).__init__(**kwargs)

    @property
    def labels(self):
        return self._labels

    def metrics(self):
        metrics = super(ClassificationExperiment, self).metrics()
        metrics.extend([
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            nnu.metrics.ConfusionMatrix(name="confusion_matrix", labels=self.labels)
        ])
        return metrics

    def train_callbacks(self):
        callbacks = super(ClassificationExperiment, self).train_callbacks()
        callbacks.extend([
            # NOTE: Once https://github.com/tensorflow/tensorboard/issues/2412 is fixed
            # set back profile_batch=2
            nnu.callbacks.ClassificationTensorBoard(log_dir=self.log_path, profile_batch=0),
        ])
        return  callbacks

    def eval_callbacks(self):
        callbacks = super(ClassificationExperiment, self).eval_callbacks()
        callbacks.extend([])
        return  callbacks
