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

from .model import models

import nnutil2 as nnu


def load_from_saved_model(saved_model_path):
    custom_objects = {}
    custom_objects.update({c.__name__: c for c in nnu.layers.layers()})
    custom_objects.update({c.__name__: c for c in nnu.models.models()})

    return tf.keras.experimental.load_from_saved_model(
        saved_model_path,
        custom_objects=custom_objects
    )

def export_saved_model(model, saved_model_path, **kwargs):
    return tf.keras.experimental.export_saved_model(
        model=model,
        saved_model_path=saved_model_path,
        **kwargs
    )
