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
    return isinstance(x, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor))


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

    elif isinstance(structure, (int, np.integer, np.signedinteger, float, np.floating, str, bytes)):
        return tf.constant(structure, shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))


def as_numpy(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_numpy, structure)

    elif is_tensor(structure):
        return structure.numpy()

    elif isinstance(structure, np.ndarray):
        return structure

    elif isinstance(structure, (int, np.integer, np.signedinteger, float, np.floating, str, bytes)):
        dtype = tf.dtype.as_dtype(type(structure)).as_numpy_dtype()
        return np.array(structure, shape=(), dtype=dtype)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
