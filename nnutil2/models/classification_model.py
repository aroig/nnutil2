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

from .model import Model

class ClassificationModel(Model):
    def __init__(self, network=None, **kwargs):
        assert network is not None
        super(ClassificationModel, self).__init__(**kwargs)
        self._network = network

    def call(self, inputs, training=False):
        return self._network(inputs, training=training)

    def get_config(self):
        config = super(ClassificationModel, self).get_config()
        config.update({
            'network': self._network.get_config()
        })
        return config

    def metrics(self):
        metrics = super(ClassificationModel, self).train_metrics()
        metrics.extend([])
        return metrics

    def train_callbacks(self):
        callbacks = super(ClassificationModel, self).train_callbacks()
        callbacks.extend([])
        return callbacks

    def train_summaries(self):
        summaries = super(ClassificationModel, self).train_summaries()
        summaries.extend([])
        return summaries

    def eval_callbacks(self):
        callbacks = super(ClassificationModel, self).eval_callbacks()
        callbacks.extend([])
        return callbacks

    def eval_summaries(self):
        summaries = super(ClassificationModel, self).eval_summaries()
        summaries.extend([])
        return summaries
