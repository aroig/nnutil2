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

class TensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoard, self).__init__(**kwargs)

        # TODO: tf.summary.create_file_writer()
