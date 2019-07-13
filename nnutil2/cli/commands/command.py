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

class Command:
    def __init__(self, experiments):
        self._experiments = experiments

    @property
    def experiments(self):
        return self._experiments

    def get_experiment(self, name):
        for exp in self.experiments:
            if exp.__name__ == name:
                return exp

        return None

    def run(argv):
        raise NotImplementedError
