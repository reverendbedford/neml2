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

import yaml
import sys
import re
from pathlib import Path


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
        value = "true" if value else "false"
    return value


def get_sections(syntax):
    sections = [params["section"] for type, params in syntax.items()]
    return list(dict.fromkeys(sections))


if __name__ == "__main__":
    outfile = Path(sys.argv[2])
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with open(sys.argv[1], "r") as stream:
        syntax = yaml.safe_load(stream)

    sections = get_sections(syntax)

    with open(sys.argv[2], "w") as stream:
        stream.write("# Syntax Documentation {#syntax}\n\n")
        stream.write("[TOC]\n\n")

        for section in sections:
            stream.write(
                "## [{}] {{#{}}}\n\n".format(section, "syntax-" + section.lower())
            )
            for type, params in syntax.items():
                if params["section"] != section:
                    continue
                input_type = demangle(params["type"]["value"])
                stream.write(
                    "### {} {{#{}}}\n\n".format(input_type, input_type.lower())
                )
                if params["doc"]:
                    stream.write("_{}_\n".format(params["doc"]))
                for param_name, info in params.items():
                    if param_name == "section":
                        continue
                    if param_name == "doc":
                        continue
                    if param_name == "name":
                        continue
                    if param_name == "type":
                        continue
                    if info["suppressed"]:
                        continue

                    param_type = demangle(info["type"])
                    param_value = postprocess(info["value"], param_type)
                    stream.write("<details>\n")
                    if not info["doc"]:
                        stream.write("  <summary>`{}`</summary>\n\n".format(param_name))
                    else:
                        stream.write(
                            "  <summary>`{}` {}</summary>\n\n".format(
                                param_name, info["doc"]
                            )
                        )
                    stream.write("  - <u>Type</u>: {}\n".format(param_type))
                    if param_value:
                        stream.write("  - <u>Default</u>: {}\n".format(param_value))
                    stream.write("</details>\n")
                stream.write("\n")
                stream.write("Detailed documentation [link](@ref {})\n\n".format(type))
