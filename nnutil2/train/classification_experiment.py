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

    def fit(self, steps_per_epoch=100, **kwargs):
        self.model.compile(metrics=self.metrics())

        batch_size = self.hyperparameters.get('batch_size', 128)
        epochs = self.hyperparameters.get('epochs', 256)
        train_steps = self.hyperparameters.get('train_steps', 1024)

        steps_per_epoch = int(train_steps / epochs)

        train_dataset, eval_dataset = self.dataset()

        # NOTE: we need to run the model in order for it to be created and do the load
        if self._resume:
            self.model.predict(x=train_dataset.take(1))
            self.load()

        return self.model.fit(
            train_dataset.batch(batch_size),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=eval_dataset.batch(batch_size),
            validation_steps=int(steps_per_epoch / 2),
            callbacks=self.train_callbacks(),
            **kwargs)

    def evaluate(self, **kwargs):
        self.model.compile(metrics=self.metrics())
        return self.model.evaluate(**kwargs)

    def predict(self, **kwargs):
        self.model.compile(metrics=[])
        return self.model.predict(**kwargs)

    def dataset(self):
        raise NotImplementedError

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
