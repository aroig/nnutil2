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


import io
import itertools

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

def image_grid():
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

    return figure

def plot_confusion_matrix(cm, labels,
                          name=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if name is None:
        name = "Confusion Matrix"

    size = len(labels)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlim=[-0.5, cm.shape[1] - 0.5],
           ylim=[-0.5, cm.shape[0] - 0.5],
           # ... and label them with the respective list entries
           xticklabels=labels,
           yticklabels=labels,
           title=name,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], ".2f"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return fig



def confusion_matrix(matrix, name='confusion_matrix', labels=None, step=None):
    assert labels is not None

    figure = plot_confusion_matrix(matrix, labels=labels)
    cm_image = plot_to_image(figure)
    summary = tf.summary.image("confusion_matrix", cm_image, step=step)
    return summary

"""
import tensorflow as tf
import numpy as np

import textwrap
import re
import io
import itertools
import matplotlib


class SaverHook(tf.train.SessionRunHook):
    def __init__(self, labels, confusion_matrix_tensor_name, summary_writer):
        self.confusion_matrix_tensor_name = confusion_matrix_tensor_name
        self.labels = labels
        self._summary_writer = summary_writer

    def end(self, session):
        cm = tf.get_default_graph().get_tensor_by_name(
                self.confusion_matrix_tensor_name + ':0').eval(session=session).astype(int)
        globalStep = tf.train.get_global_step().eval(session=session)
        figure = self._plot_confusion_matrix(cm)
        summary = self._figure_to_summary(figure)
        self._summary_writer.add_summary(summary, globalStep)

    def _figure_to_summary(self, fig):

        # attach a new canvas if not exists
        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # get PNG data from the figure
        png_buffer = io.BytesIO()
        fig.canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                      encoded_image_string=png_encoded)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.confusion_matrix_tensor_name, image=summary_image)])
        return summary

    def _plot_confusion_matrix(self, cm):
        numClasses = len(self.labels)

        fig = matplotlib.figure.Figure(figsize=(numClasses, numClasses), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(numClasses), range(numClasses)):
            ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.', horizontalalignment="center", verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig

"""
