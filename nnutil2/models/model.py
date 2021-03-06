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

_model_collection = []

def models():
    return _model_collection

def register_model(cls):
    if cls not in _model_collection:
        _model_collection.append(cls)

    return cls

class Model(tf.keras.models.Model):
    def __init__(self, hparams=None, **kwargs):
        assert hparams is not None
        self._hparams = hparams
        super(Model, self).__init__(**kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_model(cls)

    @property
    def hparams(self):
        return self._hparams

    def compile(self, metrics=[], **kwargs):
        metrics = list(metrics) + self.model_metrics()
        return super(Model, self).compile(metrics=metrics, **kwargs)

    def model_metrics(self):
        metrics = []
        return metrics

    def train_callbacks(self):
        callbacks = []
        return callbacks

    def eval_callbacks(self):
        callbacks = []
        return callbacks
