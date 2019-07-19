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


import os
from datetime import datetime

import tensorflow as tf

class Experiment:
    def __init__(self, train_path=None, data_path=None, model=None, hyperparameters={}, resume=False, seed=None):
        assert model is not None

        self._train_path = None
        if train_path is not None:
            self._train_path = os.path.join(os.path.abspath(train_path), self.name)
            if not os.path.exists(self._train_path):
                os.makedirs(self._train_path)

        self._data_path = None
        if data_path is not None:
            self._data_path = os.path.abspath(data_path)
            if not os.path.exists(self._data_path):
                os.makedirs(self._data_path)

        self._model = model
        self._hyperparameters = hyperparameters

        self._resume = resume
        self._dirname = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        if self._resume:
            self._dirname = self._last_dirname(self._dirname)

        if self.model_path is not None and not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self._seed = seed

    def _last_dirname(self, fallback):
        all_runs = []

        train_dir = self._train_path
        if os.path.exists(train_dir):
            all_runs = list(sorted(os.listdir(train_dir)))

        if len(all_runs) > 0:
            return all_runs[-1]
        else:
            return fallback

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def model_path(self):
        if self._train_path is None:
            return None

        return os.path.join(self._train_path, self._dirname, "model")

    @property
    def log_path(self):
        if self._train_path is None:
            return None

        return os.path.join(self._train_path, self._dirname, "log")

    @property
    def data_path(self):
        if self._data_path is None:
            return None

        return self._data_path

    @property
    def model(self):
        return self._model

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def compiled_model(self):
        # TODO: if not compiled
        self.model.compile(metrics=self.metrics())
        return self.model

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.model_path, "model.hdf5")

        self.model.load_weights(path)

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.model_path, "model.hdf5")

        self.model.save_weights(path)

    def metrics(self):
        metrics = []
        return metrics

    def train_callbacks(self):
        model_file = os.path.join(self.model_path, "model.hdf5")

        callbacks = [
            # NOTE: cannot use load_weights_on_restart because the model is only created upon training.
            # Before training there is no model to load weights into and the reader fails.
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_file,
                load_weights_on_restart=False
            )
        ]

        return callbacks

    def train_summaries(self):
        summaries = []
        return summaries

    def eval_callbacks(self):
        callbacks = []
        return callbacks

    def eval_summaries(self):
        summaries = []
        return summaries


_experiment_collection = []

def experiments():
    return _experiment_collection

def register_experiment(cls):
    _experiment_collection.append(cls)
    return cls
