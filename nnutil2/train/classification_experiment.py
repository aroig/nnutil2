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

from .experiment import Experiment

class ClassificationExperiment(Experiment):
    def __init__(self, labels=None, **kwargs):
        assert labels is not None
        self._labels = labels

        super(ClassificationExperiment, self).__init__(**kwargs)

    @property
    def labels(self):
        return self._labels

    def fit(self, **kwargs):
        return self.model.fit(**kwargs)

    def evaluate(self, **kwargs):
        return self.model.evaluate(**kwargs)

    def predict(self, **kwargs):
        return self.model.predict(**kwargs)

    def dataset(self):
        raise NotImplementedError

    def metrics(self):
        metrics = super(ClassificationExperiment, self).train_metrics()
        metrics.extend([])
        return metrics

    def train_callbacks(self):
        callbacks = super(ClassificationExperiment, self).train_callbacks()
        callbacks.extend([])
        return  callbacks

    def train_summaries(self):
        summaries = super(ClassificationExperiment, self).train_summaries()
        summaries.extend([])
        return summaries

    def eval_callbacks(self):
        callbacks = super(ClassificationExperiment, self).eval_callbacks()
        callbacks.extend([])
        return  callbacks

    def eval_summaries(self):
        summaries = super(ClassificationExperiment, self).eval_summaries()
        summaries.extend([])
        return summaries
