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

import os
import subprocess

class Tensorboard:
    def __init__(self, path=None):
        assert path is not None

        self._path = os.path.abspath(path)
        self._port = 6006
        self._debugger = False

    def __enter__(self):
        args = ["tensorboard", "--bind_all", "--port={}".format(self._port)]

        if self._debugger:
            args.append("--deugger_port={}".format(self._port + 1))

        args.append("--logdir={0}".format(self._path))

        self._tboard_proc = subprocess.Popen(args)
        # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return self

    def __exit__(self, type, value, traceback):
        if self._tboard_proc is not None:
            self._tboard_proc.terminate()
