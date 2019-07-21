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
            nnu.metrics.ConfusionMatrix(name="confusion_matrix", nlabels=len(self.labels)),
            nnu.metrics.F1Score(name="f1_score", nlabels=len(self.labels))
        ])

        for i, lb in enumerate(self._labels):
            pr_curve = nnu.metrics.PRCurve(label=i, name="pr_curve/{}".format(lb))
            metrics.append(pr_curve)

        return metrics

    def train_callbacks(self):
        callbacks = super(ClassificationExperiment, self).train_callbacks()
        callbacks.extend([
            # NOTE: profile_batch requires cupti properly set up. On ubuntu 18.04
            # sudo bash -c "echo /usr/local/cuda-10.0/extras/CUPTI/lib64 > /etc/ld.so.conf.d/cupti-10-0.conf"
            # sudo ldconfig
            nnu.callbacks.ClassificationTensorBoard(
                run_id=self.run_id,
                log_dir=self.log_path,
                profile_batch=2
            ),
        ])
        return  callbacks

    def eval_callbacks(self):
        callbacks = super(ClassificationExperiment, self).eval_callbacks()
        callbacks.extend([])
        return  callbacks
