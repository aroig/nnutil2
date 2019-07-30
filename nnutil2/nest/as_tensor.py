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

def as_tensor(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_tensor, structure)

    elif isinstance(structure, tf.Tensor):
        return structure

    elif isinstance(structure, np.ndarray):
        return tf.constant(structure, shape=structure.shape, dtype=tf.dtype.as_dtype(structure.dtype))

    elif any([isinstance(structure, c) for c in [int, np.int32, np.int64, float, np.float32, np.float64, str, bytes]]):
        return tf.constant(structure, shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_tensor, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
