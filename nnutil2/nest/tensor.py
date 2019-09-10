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
import numpy as np

def is_tensor(x):
    return any([isinstance(x, tf.Tensor), isinstance(x, tf.SparseTensor), isinstance(x, tf.RaggedTensor)])


def as_tensor(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_tensor, structure)

    elif isinstance(structure, tf.Tensor):
        return structure

    elif isinstance(structure, tf.SparseTensor):
        return tf.sparse.to_dense(structure)

    elif isinstance(structure, tf.RaggedTensor):
        return tf.sparse.to_dense(structure)

    elif isinstance(structure, np.ndarray):
        return tf.constant(structure, shape=structure.shape, dtype=tf.dtype.as_dtype(structure.dtype))

    elif any([isinstance(structure, c) for c in [int, np.int32, np.int64, float, np.float32, np.float64, str, bytes]]):
        return tf.constant(structure, shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_tensor, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))


def as_numpy(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_numpy, structure)

    elif is_tensor(structure):
        return structure.numpy()

    elif isinstance(structure, np.ndarray):
        return structure

    elif any([isinstance(structure, c) for c in [int, np.int32, np.int64, float, np.float32, np.float64, str, bytes]]):
        dtype = tf.dtype.as_dtype(type(structure)).as_numpy_dtype()
        return np.array(structure, shape=(), dtype=dtype)

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_numpy, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
