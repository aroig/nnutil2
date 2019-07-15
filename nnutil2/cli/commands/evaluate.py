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
import json

from nnutil2 import util
from nnutil2 import train
from .command import Command

class Evaluate(Command):
    name = "evaluate"
    description = "Evaluate an experiment"

    def __init__(self, train_path, data_path, **kwargs):
        super(Evaluate, self).__init__(**kwargs)
        self._train_path = train_path
        self._data_path = data_path

        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('-e', '--experiment', help='Experiment', required=True)

        self._parser = parser

    def run(self, argv):
        args = self._parser.parse_args(argv)

        exp_cls = self.get_experiment(args.experiment)
        exp = exp_cls(
            train_path=self._train_path,
            data_path=self._data_path
        )

        metrics = exp.evaluate()
        print(json.dumps(metrics, indent=4, sort_keys=True))
