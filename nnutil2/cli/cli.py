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

from nnutil2 import train
from . import commands

class Cli:
    def __init__(self, model_path, data_path, experiments, description=""):
        self._model_path = model_path
        self._data_path = data_path
        self._experiments = experiments

        self._description = description

        self._commands = {
            commands.Train,
            commands.PrintDataset,
            commands.Evaluate,
            commands.LsExperiments
        }

    def run(self, argv):

        parser = argparse.ArgumentParser(description=self._description)
        subparsers = parser.add_subparsers(dest="cmd", help="Commands")

        commands = {}
        for cmd in self._commands:
            commands[cmd.name] = cmd
            subparsers.add_parser(cmd.name, help=cmd.description)

        args, rest_argv = parser.parse_known_args(argv)

        cmd_cls = commands[args.cmd]
        cmd = cmd_cls(
            model_path=self._model_path,
            data_path=self._data_path,
            experiments=self._experiments

        )
        cmd.run(rest_argv)
