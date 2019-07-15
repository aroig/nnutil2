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

class LsExperiments(Command):
    name = "ls-experiments"
    description = "Print list of experiments"

    def __init__(self, train_path, data_path, **kwargs):
        super(LsExperiments, self).__init__(**kwargs)
        self._train_path = train_path
        self._data_path = data_path

        parser = argparse.ArgumentParser()

        self._parser = parser

    def run(self, argv):
        args = self._parser.parse_args(argv)

        for exp in self.experiments:
            print(exp.__name__)
