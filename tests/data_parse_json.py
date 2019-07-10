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

import unittest

import numpy as np
import tensorflow as tf
import nnutil2 as nnu

class DataParseJson(tf.test.TestCase):
    def test_data_parse_json_private(self):
        tf.random.set_seed(42)

        raw = '{"a": {"b": 1}, "c": 3}'

        tensor_spec = {
            "a": {"b": tf.TensorSpec((), tf.int32)},
            "c": tf.TensorSpec((), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nnu.data.ParseJson(ds, tensor_spec=tensor_spec)

        flat0 = ds._flatten_to_numpy({"a" : { "b": 1 }, "c": 3 })
        self.assertEqual(flat0, [np.array(1), np.array(3)])

        flat1 = ds._parse_json_fn(tf.constant(raw, dtype=tf.string))
        self.assertAllClose(flat1, [np.array(1), np.array(3)])

    def test_data_parse_json_parse_private_flatten_lists(self):
        tf.random.set_seed(42)
        raw = '{"a": [{"b": 1}, {"b": 2}], "c": 3}'

        tensor_spec = {
            "a": {"b": tf.TensorSpec((), tf.int32)},
            "c": tf.TensorSpec((), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nnu.data.ParseJson(ds, tensor_spec=tensor_spec)

        flat = ds._parse_json_flatten_lists_fn(tf.constant(raw, dtype=tf.string))
        self.assertAllClose(flat, [np.array([1, 2]), np.array([3, 3])])

    def test_data_parse_json_1(self):
        tf.random.set_seed(42)
        raw = '{"a": 1, "b": 2}'

        tensor_spec = {
            "a": tf.TensorSpec((1,), tf.int32),
            "b": tf.TensorSpec((1,), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nnu.data.ParseJson(ds, tensor_spec=tensor_spec)

        with self.cached_session() as sess:
            it = iter(ds)
            feature = sess.run(next(it))

        self.assertAllClose(feature, {'a': np.array([1]), 'b': np.array([2])})

    def test_data_parse_json_2(self):
        tf.random.set_seed(42)
        raw = '{"a": [[1, 1], [2, 2]], "b": 5}'

        tensor_spec = {
            "a": tf.TensorSpec((2,2), tf.int32),
            "b": tf.TensorSpec((1,), tf.float32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nnu.data.ParseJson(ds, tensor_spec=tensor_spec)

        with self.cached_session() as sess:
            it = iter(ds)
            feature = sess.run(next(it))

        self.assertAllClose(feature, {'a': np.array([[1, 1], [2,2]]), 'b': np.array([5.0])})

    def test_data_parse_json_3(self):
        tf.random.set_seed(42)

        raw = '{"a": [{ "b": 1 }, { "b": 2 }], "c": 3}'

        tensor_spec = {
            "a": {"b": tf.TensorSpec((), tf.int32)},
            "c": tf.TensorSpec((), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nnu.data.ParseJson(ds, tensor_spec=tensor_spec, flatten_lists=True)

        with self.cached_session() as sess:
            it = iter(ds)
            feature1 = sess.run(next(it))
            feature2 = sess.run(next(it))

        self.assertAllClose(feature1, {'a': {'b': np.array(1)}, 'c': np.array(3)})
        self.assertAllClose(feature2, {'a': {'b': np.array(2)}, 'c': np.array(3)})


if __name__ == '__main__':
    tf.test.main()
