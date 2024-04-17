#! /usr/bin/env python

# Copyright 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from math import nan
import sys
import re
import pandas as pd
import math


def time_conversion(unit):
    if unit == "m":
        return 60
    elif unit == "s":
        return 1
    elif unit == "ms":
        return 1e-3
    elif unit == "us":
        return 1e-6
    elif unit == "ns":
        return 1e-9
    else:
        return math.nan


def get_test_case(lines, i):
    multilines = lines[i]
    for line in lines[i + 1 :]:
        if line.strip() != "":
            multilines += line
        else:
            return multilines
    return multilines


def to_time(token):
    args = token.split()
    number = float(args[0])
    unit = args[1]
    return number * time_conversion(unit)


def append_config(data, token):
    config_pairs = token[1:-1].split(" ")
    for config_pair in config_pairs:
        args = config_pair.split("=")
        data.setdefault(args[0], []).append(args[1])


if __name__ == "__main__":
    test_case_lines = []

    with open(sys.argv[1]) as file:
        lines = file.readlines()
        test_case = []
        for i, line in enumerate(lines):
            if line.startswith("{"):
                test_case_lines.append(i)

    data = {
        "samples": [],
        "iterations": [],
        "estimated": [],
        "mean": [],
        "low_mean": [],
        "high_mean": [],
        "std": [],
        "low_std": [],
        "high_std": [],
    }

    for i in test_case_lines:
        case = get_test_case(lines, i)
        tokens = re.split(r"\s{2,}", case.strip())
        append_config(data, tokens.pop(0))
        data["samples"].append(tokens.pop(0))
        data["iterations"].append(tokens.pop(0))
        data["estimated"].append(to_time(tokens.pop(0)))
        data["mean"].append(to_time(tokens.pop(0)))
        data["low_mean"].append(to_time(tokens.pop(0)))
        data["high_mean"].append(to_time(tokens.pop(0)))
        data["std"].append(to_time(tokens.pop(0)))
        data["low_std"].append(to_time(tokens.pop(0)))
        data["high_std"].append(to_time(tokens.pop(0)))

    df = pd.DataFrame(data)

    df.to_csv(sys.argv[2], index=False)
