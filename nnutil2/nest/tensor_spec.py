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

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))


def get_dtype(structure):
    structure_flat = tf.nest.flatten(structure)
    dtypes = [x.dtype for x in structure_flat]

    dtype = dtypes[0]
    for dt in dtypes:
        assert dtype == dt

    return dtype


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
    ts0_spec = as_tensor_spec(ts0)
    ts1_spec = as_tensor_spec(ts1)

    try:
        tf.nest.assert_same_structure(ts0_spec, ts1_spec)
    except ValueError:
        return False

    ts0_flat = tf.nest.flatten(ts0_spec)
    ts1_flat = tf.nest.flatten(ts1_spec)

    return all([ x0 == x1 and x0.name == x1.name for x0, x1 in zip(ts0_flat, ts1_flat)])
