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
import re

def list_file_paths(path, regex=None):
    if isinstance(path, str):
        if os.path.isdir(path):
            return list_file_paths(os.listdir(path), regex=regex)

        elif os.path.exists(path):
            if regex is not None:
                if not re.search(path, regex):
                    return []
            return [path]

    elif isinstance(path, list):
        return [p for q in path for p in list_file_paths(q, regex=regex)]

    else:
        raise Exception("Cannot handle path type: {}".format(type(path)))
