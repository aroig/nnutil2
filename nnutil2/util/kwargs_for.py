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

import inspect

import tensorflow as tf

def kwargs_for(kwargs, func):
    sig = [p.name for p in inspect.signature(func).parameters.values()]
    args = {k: kwargs[k] for k in set(sig) & set(kwargs.keys())}
    return args
