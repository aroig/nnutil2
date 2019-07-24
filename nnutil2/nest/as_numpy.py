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

def as_numpy(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_numpy, structure)

    elif isinstance(structure, tf.Tensor):
        return structure.numpy()

    elif isinstance(structure, np.ndarray):
        return structure

    elif type(structure) in set([int, float, str]):
        dtype = tf.dtype.as_dtype(type(structure)).as_numpy_dtype()
        return tf.array(structure, shape=(), dtype=dtype)

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_numpy, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
