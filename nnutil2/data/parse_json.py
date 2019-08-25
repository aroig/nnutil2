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
import json
import itertools

import tensorflow as tf
import numpy as np

from tensorflow.python.util import nest

class ParseJson(tf.data.Dataset):
    def __init__(self, dataset, tensor_spec=None, flatten_lists=False):
        assert tensor_spec is not None

        self._input_datasets = [dataset]
        self._tensor_spec = tensor_spec

        self._json_flat_shapes = [spec.shape for spec in tf.nest.flatten(self._tensor_spec)]
        self._input_shapes = tf.nest.pack_sequence_as(self._tensor_spec, self._json_flat_shapes)

        self._json_flat_types = [spec.dtype for spec in tf.nest.flatten(self._tensor_spec)]
        self._input_types = tf.nest.pack_sequence_as(self._tensor_spec, self._json_flat_types)

        if flatten_lists:
            dataset = dataset.flat_map(self._parse_json_flat)
        else:
            dataset = dataset.flat_map(self._parse_json_nested)

        self._dataset = dataset

        super(ParseJson, self).__init__(self._dataset._variant_tensor)

    def _inputs(self):
        return list(self._input_datasets)

    @property
    def element_spec(self):
        return self._dataset.element_spec

    def _filter_keys(self, data, input_types):
        """Only keep keys in the nested structure data, that match an entry in input_types
        """
        if type(data) == tuple and type(input_types) == tuple:
            return tuple([self._filter_keys(x, t) for x, t in zip(data, input_types)])

        elif type(data) == dict and type(input_types) == dict:
            return {k: self._filter_keys(data[k], t) for k, t in input_types.items()}

        elif type(data) == list and type(input_types) == dict:
            return [self._filter_keys(x, input_types) for x in data]

        elif type(data) == list and type(input_types) == tuple:
            return [self._filter_keys(x, input_types) for x in data]

        else:
            return data

    def _flatten_lists(self, data, input_types):
        """Remove lists in inner nodes of a nested structure, by instantiating multiple
           nested structures replacing the lists with every possible combination of its elements
        """

        if type(data) == tuple and type(input_types) == tuple:
            flat_inner = [self._flatten_lists(x, t) for x, t in zip(data, input_types)]
            return list(itertools.product(*flat_inner))

        elif type(data) == dict and type(input_types) == dict:
            flat_inner = [[(k, xi) for xi in self._flatten_lists(data[k], t)]
                          for k, t in input_types.items()]
            return [dict(tup) for tup in itertools.product(*flat_inner)]

        elif type(data) == list and type(input_types) == dict:
            flat_inner = [self._flatten_lists(x, input_types) for x in data]
            return list(itertools.chain(*flat_inner))

        elif type(data) == list and type(input_types) == tuple:
            flat_inner = [self._flatten_lists(x, input_types) for x in data]
            return list(itertools.chain(*flat_inner))

        else:
            return [data]

    def _flatten_to_numpy(self, data):
        flat_data = nest.flatten_up_to(self._input_types, data)
        return [np.reshape(np.array(x, dtype=t.as_numpy_dtype), s)
                for x, s, t in zip(flat_data, self._json_flat_shapes, self._json_flat_types)]

    def _parse_json_fn(self, raw):
        data = json.loads(raw.numpy())
        data = self._filter_keys(data, self._input_types)
        return self._flatten_to_numpy(data)

    def _parse_json_flatten_lists_fn(self, raw):
        data = json.loads(raw.numpy())
        data = self._filter_keys(data, self._input_types)

        data_list = self._flatten_lists(data, self._input_types)
        flat_data = [self._flatten_to_numpy(data) for data in data_list]

        return [np.stack(xtuple) for xtuple in zip(*flat_data)]

    def _parse_json_flat(self, raw):
        flat_feature = tf.py_function(
            self._parse_json_flatten_lists_fn,
            [raw],
            self._json_flat_types)

        feature = tf.nest.pack_sequence_as(self._input_types, flat_feature)

        return tf.data.Dataset.from_tensor_slices(feature)

    def _parse_json_nested(self, raw):
        flat_feature = tf.py_function(
            self._parse_json_fn,
            [raw],
            self._json_flat_types)

        feature = tf.nest.pack_sequence_as(self._input_types, flat_feature)

        return tf.data.Dataset.from_tensors(feature)
