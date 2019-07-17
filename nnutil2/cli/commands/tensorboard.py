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

class Tensorboard(Command):
    name = "tensorboard"
    description = "Start tensorboard on experiment data"

    def __init__(self, train_path, data_path, **kwargs):
        super(Tensorboard, self).__init__(**kwargs)
        self._train_path = train_path
        self._data_path = data_path

        parser = argparse.ArgumentParser(description=self.description)
        self._parser = parser

    def run(self, argv):
        args = self._parser.parse_args(argv)

        with util.Tensorboard(path=self._train_path):
            while True:
                time.sleep(10)
