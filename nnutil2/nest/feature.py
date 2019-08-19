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

from .tensor_spec import as_tensor_spec

def as_feature(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_feature, structure)

    elif isinstance(structure, tf.Tensor):
        return as_feature(structure.numpy())

    elif isinstance(structure, np.ndarray):
        flat = structure.flatten().tolist()

        if any([structure.dtype == dt for dt in [int, np.int32, np.int64]]):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=flat))

        elif any([structure.dtype == dt for dt in [float, np.float32, np.float64]]):
            return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

        else:
            raise Exception("Unhandled array type: {}".format(structure.dtype))

    elif isinstance(structure, bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[structure]))

    elif isinstance(structure, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[structure.encode()]))

    elif any([isinstance(structure, c) for c in [float, np.float32, np.float64]]):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[structure]))

    elif any([isinstance(structure, c) for c in [int, np.int32, np.int64]]):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[structure]))

    elif any([isinstance(structure, c) for c in [tf.data.experimental.NestedStructure,
                                                 tf.data.experimental.TensorStructure,
                                                 tf.data.experimental.SparseTensorStructure]]):
        return as_feature(as_tensor_spec(structure))

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))


def as_feature_spec(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_feature_spec, structure)

    elif isinstance(structure, tf.Tensor):
        if structure.shape.is_fully_defined():
            return tf.io.FixedLenFeature(shape=structure.shape, dtype=structure.dtype)

        elif structure.shape.rank is not None and structure.shape[1:].is_fully_defined():
            return tf.io.FixedLenSequenceFeature(shape=structure.shape[1:], allow_missing=True, dtype=structure.dtype)

        else:
            return tf.io.VarLenFeature(dtype=structure.dtype)

    elif isinstance(structure, tf.TensorSpec):
        if structure.shape.is_fully_defined():
            return tf.io.FixedLenFeature(shape=structure.shape, dtype=structure.dtype)

        elif structure.shape.rank is not None and structure.shape[1:].is_fully_defined():
            return tf.io.FixedLenSequenceFeature(shape=structure.shape[1:], allow_missing=True, dtype=structure.dtype)

        else:
            return tf.io.VarLenFeature(dtype=structure.dtype)

    elif any([isinstance(structure, c) for c in [tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature, tf.io.VarLenFeature]]):
        return structure

    elif isinstance(structure, np.ndarray):
        return tf.io.FixedLenFeature(shape=structure.shape, dtype=tf.dtype.as_dtype(structure.dtype))

    elif any([isinstance(structure, c) for c in [int, np.int32, np.int64]]):
        return tf.io.FixedLenFeature(shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif any([isinstance(structure, c) for c in [float, np.float32, np.float64]]):
        return tf.io.FixedLenFeature(shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif any([isinstance(structure, c) for c in [str, bytes]]):
        return tf.io.FixedLenFeature(shape=(), dtype=tf.dtype.as_dtype(type(structure)))

    elif any([isinstance(structure, c) for c in [tf.data.experimental.NestedStructure,
                                                 tf.data.experimental.TensorStructure,
                                                 tf.data.experimental.SparseTensorStructure]]):
        return as_feature_spec(as_tensor_spec(structure))

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
