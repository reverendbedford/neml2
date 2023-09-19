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
import yaml
import sys
import re


def get_type(params):
    for param in params:
        for param_name, info in param.items():
            if param_name == "type":
                return demangle(info["value"])


def demangle(type):
    type = type.replace(
        "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >",
        "std::string",
    )
    type = re.sub(", std::allocator<.+> ", "", type)
    type = type.replace("neml2::", "")
    type = type.replace("std::", "")
    type = type.replace("at::", "")
    type = re.sub("CrossRef<(.+)>", r"\1", type)
    return type


def postprocess(value, type):
    if type == "bool":
        value = str(bool(value))
    return value


with open("syntax.yml", "r") as stream:
    syntax = yaml.safe_load(stream)

with open(sys.argv[1], "w") as stream:
    stream.write("# Syntax Documentation {#syntax}\n\n")
    stream.write("[TOC]\n\n")
    for s in syntax:
        for type, params in s.items():
            input_type = get_type(params)
            stream.write("## {}\n\n".format(input_type))
            names = []
            types = []
            values = []
            for param in params:
                for param_name, info in param.items():
                    if param_name == "name":
                        continue
                    if param_name == "type":
                        continue
                    param_type = demangle(info["type"])
                    param_value = postprocess(info["value"], param_type)
                    stream.write("- {}\n".format(param_name))
                    stream.write("  - **Type**: {}\n".format(param_type))
                    if param_value != None:
                        stream.write("  - **Default**: {}\n".format(param_value))
            stream.write("\n")
            stream.write("Details: [{}](@ref {})\n\n".format(input_type, type))
