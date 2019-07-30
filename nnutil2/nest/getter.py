#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - tensorflow utilities for training neural networks
# copyright (c) 2019, abdó roig-maranges <abdo.roig@gmail.com>
#
# this file is part of 'nnutil2'.
#
# this file may be modified and distributed under the terms of the 3-clause bsd
# license. see the license file for details.


def getter(key):
    """Create a getter for a nested structure, given either a key string, or a lambda"""
    if isinstance(key, str):
        return lambda x: x[key]

    elif hasattr(key, "__call__"):
        return lambda x: key(nested)

    else:
        raise Exception("Cannot access nested structure member")
