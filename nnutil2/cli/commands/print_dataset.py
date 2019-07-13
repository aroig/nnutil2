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

import argparse

from nnutil2 import train

from .command import Command

class PrintDataset(Command):
    def __init__(self, model_path, data_path):
        self._model_path = model_path
        self._data_path = data_path

        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--experiment', help='Experiment')

        self._parser = parser

    def run(self, argv):
        args = self._parser.parse_args(argv)

        exp_cls = train.get_experiment(args.experiment)
        exp = exp_cls(model_path=self._model_path, data_path=self._data_path)

        dataset = exp.dataset()
        for x in dataset:
            print(x)
