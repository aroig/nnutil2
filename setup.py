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


from setuptools import setup, find_packages
from nnutil2 import __version__

setup(
    name             = 'nnutil2',
    version          = __version__,
    license          = 'BSD',
    description      = 'Tensorflow utilities for training neural networks',
    author           = 'Abdó Roig-Maranges',
    author_email     = 'abdo.roig@gmail.com',
    packages         = find_packages(),
    install_requires = [
    ]
)
