#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2020, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

import tensorflow as tf

from .layer import Layer

from ..util import as_shape, normalize_axis, complementary_axis, num_elements, restrict_shape

class ReplicaConv(Layer):
    """
    Dense layer replicated along a dimension
    """
    def __init__(self,
                 nfilters=None,
                 kernel_size=None,
                 strides=None,
                 replica_axes=None,
                 convolution_axes=None,
                 channel_axes=None,
                 activation=None,
                 residual=False,
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 kernel_initializer=None, bias_initializer=None,
                 dtype=tf.float32,
                 *args, **kwargs):

        assert nfilters is not None
        assert kernel_size is not None

        assert replica_axes is not None
        assert convolution_axes is not None
        assert channel_axes is not None

        if strides is None:
            strides = len(kernel_size) * (1,)

        assert len(convolution_axes) == len(kernel_size)
        assert len(kernel_size) == len(strides)
        assert len(nfilters) == len(channel_axes)

        self._dtype = dtype

        self._nfilters = as_shape(nfilters)
        self._kernel_size = kernel_size
        self._strides = strides

        self._convolution_axes = convolution_axes
        self._replica_axes = replica_axes
        self._channel_axes = channel_axes
        self._conv_dim = len(convolution_axes)

        self._activation = tf.keras.activations.get(activation)
        self._residual = residual

        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)

        super(ReplicaConv, self).__init__()

    def _approx_id(self, shape):
        kernel_shape = as_shape(self._kernel_size)
        size = kernel_shape.num_elements()

        in_channels = shape[-1]
        out_channels = shape[-2]

        assert in_channels == out_channels * size

        eye = tf.reshape(tf.eye(out_channels), shape=(out_channels, out_channels, 1))
        onehot = tf.reshape(tf.one_hot(size // 2, size), shape=(1, 1, size))

        ret = tf.reshape(eye * onehot, shape=(1, out_channels, in_channels))
        return ret

    def build(self, input_shape):
        replica_axes = normalize_axis(input_shape, self._replica_axes)
        convolution_axes = normalize_axis(input_shape, self._convolution_axes)
        channel_axes = normalize_axis(input_shape, self._channel_axes)

        kernel_shape = as_shape(self._kernel_size)

        assert len(set(convolution_axes + channel_axes) & set(replica_axes)) == 0
        assert len(self._nfilters) == len(channel_axes)

        replica_dim = num_elements(input_shape, replica_axes)
        contraction_dim = num_elements(input_shape, channel_axes) * kernel_shape.num_elements()
        nfilters_dim = num_elements(self._nfilters)

        def residual_initializer(shape, dtype):
            x = self._kernel_initializer(shape, dtype)
            if self._residual:
                x = x + self._approx_id(tf.TensorShape(shape))
            return x

        self._weights = self.add_weight(
            name="weights",
            shape=(replica_dim, nfilters_dim, contraction_dim),
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint,
            initializer=residual_initializer,
            dtype=self._dtype
        )

        self._bias = self.add_weight(
            name="bias",
            shape=(replica_dim, nfilters_dim, 1),
            regularizer=self._bias_regularizer,
            constraint=self._bias_constraint,
            initializer=self._bias_initializer,
            dtype=self._dtype
        )

    def get_config(self):
        config = {
            "nfilters": self._nfilters,
            "replica_axes": self._replica_axes,
            "contraction_axes": self._contraction_axes,
            "activation": self._activation,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "kernel_constraint": self._kernel_constraint,
            "bias_cosntraint": self._bias_constraint,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "dtype": self._dtype
        }

        base_config = super(ReplicaDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _conv(self, x):
        kernel_shape = as_shape(self._kernel_size)
        size = kernel_shape.num_elements()
        kernel = tf.reshape(tf.eye(size), shape=kernel_shape + (1, size))

        # x.shape = (batch, 1, conv)
        assert x.shape.rank == kernel_shape.rank + 2

        if kernel_shape.rank == 1:
            y = tf.nn.conv1d(x, kernel, stride=self._strides, padding='SAME', data_format='NCW')

        elif kernel_shape.rank == 2:
            y = tf.nn.conv2d(x, kernel, strides=self._strides, padding='SAME', data_format='NCHW')

        elif kernel_shape.rank == 3:
             y = tf.nn.conv3d(x, kernel, strides=self._strides, padding='SAME', data_format='NCDHW')

        else:
            raise Exception("Unhandled convolution rank")

        # y.shape: (batch, size, conv // strides)
        return y

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape

        kernel_shape = as_shape(self._kernel_size)

        convolution_axes = normalize_axis(input_shape, self._convolution_axes)
        channel_axes = normalize_axis(input_shape, self._channel_axes)
        replica_axes = normalize_axis(input_shape, self._replica_axes)

        complementary_axes = complementary_axis(input_shape, channel_axes + convolution_axes + replica_axes)
        assert(len(complementary_axes) == 0)

        assert max(replica_axes) < min(channel_axes)
        assert max(replica_axes + channel_axes) < min(convolution_axes)

        # Now the indices are clustered as follows:
        # replica_axes, channel_axes, convolutional_axes

        flat_shape_0 = tf.TensorShape([
            num_elements(input_shape, replica_axes + channel_axes),
            1
        ]) + restrict_shape(input_shape, convolution_axes)

        assert flat_shape_0.num_elements() == input_shape.num_elements()

        x = tf.reshape(inputs, shape=flat_shape_0)
        y = self._conv(x)

        convolutional_shape = tf.TensorShape([
            input_shape[a] // self._strides[i] for i, a in enumerate(convolution_axes)
        ])

        flat_shape_1 = tf.TensorShape([
            num_elements(input_shape, replica_axes),
            num_elements(input_shape, channel_axes) * kernel_shape.num_elements(),
            num_elements(convolutional_shape),
        ])

        assert flat_shape_1.num_elements() == y.shape.num_elements()
        y = tf.reshape(y, shape=flat_shape_1)

        z = tf.linalg.matmul(self._weights, y) + self._bias

        output_shape = self.compute_output_shape(input_shape)
        assert output_shape.num_elements() == z.shape.num_elements()

        z = tf.reshape(z, shape=output_shape)
        z = self._activation(z)

        return z

    def compute_output_shape(self, input_shape):
        convolution_axes = normalize_axis(input_shape, self._convolution_axes)
        channel_axes = normalize_axis(input_shape, self._channel_axes)
        output_shape = input_shape.as_list()

        for i, a in enumerate(convolution_axes):
            output_shape[a] = output_shape[a] // self._strides[i]

        for i, a in enumerate(channel_axes):
            output_shape[a] = self._nfilters[i]

        return tf.TensorShape(output_shape)
