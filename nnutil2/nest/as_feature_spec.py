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

def as_feature_spec(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_feature_spec, structure)

    elif isinstance(structure, tf.Tensor):
        if structure.shape.is_fully_defined():
            return tf.io.FixedLenFeature(shape=structure.shape, dtype=structure.dtype)

        elif structure.shape.rank is not None and structure.shape[1:].is_fully_defined():
            return tf.io.FixedLenSequenceFeature(shape=structure.shape[1:], dtype=structure.dtype)

        else:
            return tf.io.VarLenFeature(dtype=structure.dtype)

    elif isinstance(structure, tf.TensorSpec):
        return tf.io.FixedLenFeature(shape=structure.shape, dtype=structure.dtype)

    elif type(structure) in set([tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature, tf.io.VarLenFeature]):
        return structure

    elif isinstance(structure, np.ndarray):
        return tf.io.FixedLenFeature(shape=structure.shape, dtype=tf.dtype.as_dtype(structure.dtype))

    elif type(structure) in set([int, np.int32, np.int64, float, np.float32, np.float64, str, bytes]):
        return tf.io.FixedLenFeature(shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_feature_spec, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
