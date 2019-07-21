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
import nnutil2 as nnu

def confusion_matrix(matrix, name='confusion_matrix', labels=None, step=None):
    assert labels is not None

    figure = nnu.visualization.plot_confusion_matrix(matrix, labels=labels)
    cm_image = nnu.visualization.figure_to_image(figure)
    summary = tf.summary.image("confusion_matrix", cm_image, step=step)
    return summary
