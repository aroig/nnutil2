#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - tensorflow utilities for training neural networks
# copyright (c) 2019, abdó roig-maranges <abdo.roig@gmail.com>
#
# this file is part of 'nnutil2'.
#
# this file may be modified and distributed under the terms of the 3-clause bsd
# license. see the license file for details.


import tensorflow as tf
import numpy as np

def as_tensor_spec(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_tensor_spec, structure)

    elif isinstance(structure, tf.Tensor):
        return tf.TensorSpec(shape=structure.shape, dtype=structure.dtype)

    elif isinstance(structure, tf.TensorSpec):
        return structure

    elif isinstance(structure, np.ndarray):
        return tf.TensorSpec(shape=structure.shape, dtype=structure.dtype)

    elif type(structure) in set([int, float, str]):
        return tf.TensorSpec(shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_tensor_spec, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
