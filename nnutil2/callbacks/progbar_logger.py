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


class ProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, logged_metrics=None):
        self._logged_metrics = logged_metrics
        self.verbose = 1

        super(ProgbarLogger, self).__init__(
            count_mode='steps',
            stateful_metrics=logged_metrics
        )

    def on_train_begin(self, logs=None):
        self.params['metrics'] = [m for m in self.params['metrics']
                                  if m in self._logged_metrics]
        self.epochs = self.params['epochs']
