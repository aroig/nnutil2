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

def getter(key):
    """Create a getter for a nested structure, given either a key string, or a lambda"""
    if type(key) == str:
        return lambda x: x[key]

    elif hasattr(key, "__call__"):
        return lambda x: key(nested)

    else:
        raise Exception("Cannot access nested structure member")
