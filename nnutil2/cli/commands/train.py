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
import time

from nnutil2 import util
from nnutil2 import train
from .command import Command

class Train(Command):
    name = "train"
    description = "Train an experiment"

    def __init__(self, train_path, data_path, **kwargs):
        super(Train, self).__init__(**kwargs)
        self._train_path = train_path
        self._data_path = data_path

        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('-e', '--experiment', help='Experiment', required=True)

        parser.add_argument('--resume', action='store_true', help='Resume previous training')
        parser.add_argument('--tracing', action='store_true', help='Produce tracing data')
        parser.add_argument('--debug', action='store_true', help='Start debugging session')

        self._parser = parser

    def run(self, argv):
        args = self._parser.parse_args(argv)

        exp_cls = self.get_experiment(args.experiment)
        exp = exp_cls(
            train_path=self._train_path,
            data_path=self._data_path
        )

        with util.Tensorboard(path=self._train_path):
            exp.fit()

            print("Training has finished!")
            while True:
                time.sleep(10)
