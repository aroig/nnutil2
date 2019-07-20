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

def make_spec(structure):
    """Construct a tensor specification from either a tensor specification or a nested structure of tensors
    """

    def func(x):
        if isinstance(x, tf.TensorSpec):
            return tf.TensorSpec(shape=x.shap, dtype=x.dtype)

        elif isinstance(x, tf.Tensor):
            return tf.TensorSpec(shape=x.shape, dtype=x.dtype)

        else:
            raise Exception("Cannot obtain a tensor spec from an object of type: {}".format(type(x)))

    return tf.nest.map_structure(func, structure)
