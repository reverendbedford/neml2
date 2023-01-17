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
    if unit == "s":
        return 1e6
    elif unit == "ms":
        return 1e3
    elif unit == "us":
        return 1
    elif unit == "ns":
        return 1e-3
    else:
        return math.nan


def get_multilines(lines, i):
    multilines = lines[i]
    for line in lines[i+1:]:
        if line.strip() != "":
            multilines += line
        else:
            return multilines


def pop_size(tokens):
    token = tokens.pop(0)
    return int(token[1:-1])


def pop_name(tokens):
    name = tokens.pop(0)
    while not tokens[0].isnumeric():
        name += tokens.pop(0)
    return name


def pop_time(tokens):
    number = tokens.pop(0)
    unit = tokens.pop(0)
    return float(number)*time_conversion(unit)


benchmarks = {}

with open(sys.argv[1]) as file:
    lines = file.readlines()
    test_case = []
    for i, line in enumerate(lines):
        if line.startswith("-") and lines[i+4].startswith("."):
            test_case_name = lines[i+1].strip()
            benchmarks[test_case_name] = []
            continue
        if re.match("\\{[0-9]+\\}", line):
            benchmarks[test_case_name].append(i)

data = {"test_case": [], "benchmark": [], "nbatch": [], "chunk_size": [], "samples": [], "iterations": [],
        "estimated": [], "mean": [], "low_mean": [], "high_mean": [], "std": [], "low_std": [], "high_std": []}

for test_case_name, indices in benchmarks.items():
    for i in indices:
        multilines = get_multilines(lines, i)
        tokens = multilines.split()
        data["test_case"].append(test_case_name)
        data["nbatch"].append(pop_size(tokens))
        data["chunk_size"].append(pop_size(tokens))
        data["benchmark"].append(pop_name(tokens))
        data["samples"].append(int(tokens.pop(0)))
        data["iterations"].append(int(tokens.pop(0)))
        data["estimated"].append(pop_time(tokens))
        data["mean"].append(pop_time(tokens))
        data["low_mean"].append(pop_time(tokens))
        data["high_mean"].append(pop_time(tokens))
        data["std"].append(pop_time(tokens))
        data["low_std"].append(pop_time(tokens))
        data["high_std"].append(pop_time(tokens))

df = pd.DataFrame(data)

df.to_csv(sys.argv[2])
