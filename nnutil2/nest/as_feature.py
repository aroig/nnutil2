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

def as_feature(structure):
    if tf.nest.is_nested(structure):
        return tf.nest.map_structure(as_feature, structure)

    elif isinstance(structure, tf.Tensor):
        return as_feature(structure.numpy())

    elif isinstance(structure, np.ndarray):
        flat = structure.flatten().tolist()

        if structure.dtype in set([int, np.int32, np.int64]):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=flat))

        elif structure.dtype in set([float, np.float32, np.float64]):
            return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

        else:
            raise Exception("Unhandled array type: {}".format(structure.dtype))

    elif type(structure) in set([bytes]):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[structure]))

    elif type(structure) in set([str]):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[structure.encode()]))

    elif type(structure) in set([float, np.float32, np.float64]):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[structure]))

    elif type(structure) in set([int, np.int32, np.int64]):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[structure]))

    elif isinstance(structure, tf.data.experimental.NestedStructure):
        return tf.nest.map_structure(as_tensor, structure._nested_structure)

    else:
        raise Exception("Cannot handle nested structure of type: {}".format(type(structure)))
