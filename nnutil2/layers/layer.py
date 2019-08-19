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
import tensorflow.keras as ks

_layer_collection = []

def layers():
    return _layer_collection

def register_layer(cls):
    if cls not in _layer_collection:
        _layer_collection.append(cls)

class Layer(ks.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_layer(cls)
