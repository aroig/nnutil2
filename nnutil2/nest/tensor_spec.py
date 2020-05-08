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


import numpy as np
import tensorflow as tf

def as_tensor_spec(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_tensor_spec, structure)

    elif isinstance(structure, tf.Tensor):
        return tf.TensorSpec(shape=structure.shape, dtype=structure.dtype)

    elif isinstance(structure, tf.TensorSpec):
        return structure

    elif isinstance(structure, np.ndarray):
        return tf.TensorSpec(shape=structure.shape, dtype=structure.dtype)

    elif isinstance(structure, (int, np.integer, np.signedinteger, float, np.floating, str, bytes)):
        return tf.TensorSpec(shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif isinstance(structure, tf.data.experimental.TensorStructure):
        return tf.TensorSpec(shape=structure.shape, dtype=structure.dtype)

    elif isinstance(structure, tf.data.experimental.SparseTensorStructure):
        return tf.TensorSpec(shape=structure.dense_shape, dtype=structure.dtype)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))


def tensor_spec_to_python(tensor_spec):
    """Converts a nested TensorSpec into a python nested dict that can be json serialized"""

    if tf.nest.is_nested(tensor_spec):
        return tf.nest.map_structure(tensor_spec_to_python, tensor_spec)

    elif isinstance(tensor_spec, tf.TensorSpec):
        return {
            'name': tensor_spec.name,
            'shape': tensor_spec.shape.as_list(),
            'dtype': tensor_spec.dtype.name,
        }

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(tensor_spec)))


def python_to_tensor_spec(tensor_spec):
    """Converts python nested structure to a TensorSpec nested structure"""

    if isinstance(tensor_spec, dict) and set(tensor_spec.keys()) == set(['name', 'shape', 'dtype']):
        name = tensor_spec['name']
        shape = tf.TensorShape(tensor_spec['shape'])
        dtype = tf.dtypes.as_dtype(tensor_spec['dtype'])
        return tf.TensorSpec(name=name, shape=shape, dtype=dtype)

    elif isinstance(tensor_spec, dict):
        return {k : python_to_tensor_spec(v) for k, v in tensor_spec.items() }

    elif isinstance(tensor_spec, list):
        return [python_to_tensor_spec(v) for v in tensor_spec]

    elif isinstance(tensor_spec, tuple):
        return tuple([python_to_tensor_spec(v) for v in tensor_spec])

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(tensor_spec)))


def same_tensor_spec(ts0, ts1):
    if tf.nest.is_nested(ts0) and tf.nest.is_nested(ts1):
        try:
            tf.nest.assert_same_structure(ts0, ts1)
        except ValueError:
            return False

        ts0_flat = tf.nest.flatten(ts0)
        ts1_flat = tf.nest.flatten(ts1)

        return all([same_tensor_spec(x0, x1) for x0, x1 in zip(ts0_flat, ts1_flat)])

    elif isinstance(ts0, tf.TensorSpec) and isinstance(ts1, tf.TensorSpec):
        return ts0 == ts1 and ts0.name == ts1.name

    return False
