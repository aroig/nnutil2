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

import nnutil2 as nnu

_experiment_collection = []

def experiments():
    return _experiment_collection

def register_experiment(cls):
    if cls not in _experiment_collection:
        _experiment_collection.append(cls)

    return cls

class Experiment:
    def __init__(self, train_path=None, data_path=None, model=None, hparams={},
                 validation_steps=None, resume=False, seed=None):
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

        self._compiled = False
        self._model = model

        assert 'batch_size' in hparams
        assert 'train_steps' in hparams

        self._hparams = hparams

        if validation_steps is None:
            validation_steps = 8
        self._validation_steps = validation_steps

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
    def run_id(self):
        return "{}/{}".format(self.name, self._dirname)

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
    def export_path(self):
        if self._train_path is None:
            return None

        return os.path.join(self._train_path, self._dirname, "export")

    @property
    def model(self):
        if not self._compiled:
            self._model.compile(metrics=self.metrics())
            self._compiled = True

        return self._model

    @property
    def input_signature(self):
        train_dataset = self.train_dataset(repeat=True)

        if train_dataset is not None:
            return nnu.nest.as_tensor_spec(train_dataset._element_structure)

        return None

    @property
    def hparams(self):
        return self._hparams

    def eval_dataset(self, repeat=False):
        return None

    def train_dataset(self, repeat=False):
        return None

    def fit(self, epochs=32, **kwargs):
        batch_size = self.hparams.get('batch_size', 128)
        train_steps = self.hparams.get('train_steps', 1024)

        steps_per_epoch = int(train_steps / epochs)

        train_dataset = self.train_dataset(repeat=True)
        if train_dataset is not None:
            train_dataset = train_dataset.batch(batch_size)

        eval_dataset = self.eval_dataset(repeat=True)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.batch(batch_size)

        if self._resume:
            self.load()

        return self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=eval_dataset,
            validation_steps=self._validation_steps,
            callbacks=self.train_callbacks(),
            verbose=0,
            **kwargs)

    def evaluate(self, **kwargs):
        batch_size = self.hparams.get('batch_size', 128)

        eval_dataset = self.eval_dataset()
        if eval_dataset is not None:
            eval_dataset = eval_dataset.batch(batch_size)

        metrics = self.model.evaluate(
            eval_dataset,
            steps=self._validation_steps,
            callbacks=self.eval_callbacks(),
            **kwargs)

        metrics_dict = {}
        for name, value in zip(self.model.metrics_names, metrics):
            metrics_dict[name] = value

        return metrics_dict

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.model_path, "model.hdf5")

        if not self.model.inputs:
            self.model._set_inputs(nnu.nest.as_tensor_spec(self.input_signature), training=False)

        self.model.load_weights(path)

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.model_path, "model.hdf5")

        self.model.save_weights(path)

    def export(self, path=None):
        if path is None:
            path = self.export_path

        if not self.model.inputs:
            self.model._set_inputs(nnu.nest.as_tensor_spec(self.input_signature), training=False)

        nnu.models.export_saved_model(
            self.model,
            path,
            input_signature=[self.input_signature],
            serving_only=True,
        )

    def dataset_stats(self):
        raise NotImplementedError

    def metrics(self):
        metrics = []
        return metrics

    def progbar_metrics(self):
        metrics = ["loss", "val_loss"]
        return metrics

    def train_callbacks(self):
        model_file = os.path.join(self.model_path, "model.hdf5")
        progbar_metrics = self.progbar_metrics()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_file,
            ),
            nnu.callbacks.ProgbarLogger(
                logged_metrics=progbar_metrics
            )
        ]

        return callbacks

    def eval_callbacks(self):
        callbacks = []
        return callbacks
