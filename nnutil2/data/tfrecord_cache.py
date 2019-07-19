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


import os

import numpy as np
import tensorflow as tf


class TFRecordCache(tf.data.Dataset):
    def __init__(self, dataset, path):
        if self.needs_rebuild(path):
            print("Preparing tfrecord: {}".format(path))

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            writer = tf.io.TFRecordWriter(path)
            for x in dataset:
                example = self.serialize_example(x)
                writer.write(example)

        self._tensor_spec = dataset._element_structure._nested_structure

        if not os.path.exists(path):
            raise Exception("tfrecord file does not exist: {}".format(path))

        tfrecord_dataset = tf.data.TFRecordDataset(path)
        self._dataset = tfrecord_dataset.map(self.parse_example)

        super(TFRecordCache, self).__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return []

    @property
    def _element_structure(self):
        return tf.data.experimental.NestedStructure(self._tensor_spec)

    def serialize_example(self, x):
        example = tf.train.Example(features=self.make_feature(x))
        return example.SerializeToString()

    def make_feature(self, feature):
         if type(feature) == dict:
             return tf.train.Features(feature={k: self.make_feature(v) for k, v in feature.items()})

         elif type(feature) == bytes:
             return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))

         elif type(feature) in set([float, np.float32, np.float64]):
             return tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))

         elif type(feature) in set([int, np.int32, np.int64]):
             return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))

         elif type(feature) in set([np.array, np.ndarray]):
             flat = feature.flatten().tolist()

             if feature.dtype == int or feature.dtype == np.int32 or feature.dtype == np.int64:
                 return tf.train.Feature(int64_list=tf.train.Int64List(value=flat))

             elif feature.dtype == float or feature.dtype == np.float32 or feature.dtype == np.float64:
                 return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

             else:
                 raise Exception("Unhandled array type: {}".format(feature.dtype))

         elif type(feature) in set([type(tf.constant(0))]):
             return self.make_feature(feature.numpy())

         else:
             raise Exception("Unhandled feature type: {}".format(type(feature)))

    def parse_spec(self, tensor_spec):
        if type(tensor_spec) == dict:
            return {k: self.parse_spec(v) for k, v in tensor_spec.items()}

        elif type(tensor_spec) == tf.TensorSpec:
            return tf.io.FixedLenFeature(tensor_spec.shape, tensor_spec.dtype)

        elif type(tensor_spec) == tf.io.FixedLenFeature:
            return tensor_spec

        elif type(tensor_spec) == tf.io.VarLenFeature:
            return tensor_spec

        else:
            raise Exception("Unhandled input spec: {}".format(type(tensor_spec)))

    def parse_example(self, example_proto):
        parse_spec = self.parse_spec(self._tensor_spec)
        parsed_features = tf.io.parse_single_example(example_proto, parse_spec)

        return parsed_features

    def needs_rebuild(self, path):
        return not os.path.exists(path)
