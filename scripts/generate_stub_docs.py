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

from pathlib import Path

rootdir = Path(".")

outdir = Path("doc/sphinx/source")

file_list = [f.stem for f in rootdir.resolve().glob("include/**/*.h") if f.is_file()]

outdir.mkdir(parents=True, exist_ok=True)

template = """{0}
=========================================================================

.. doxygenfile:: {0}.h
"""

for f in file_list:
    ofile = Path(outdir / f).with_suffix(".rst")
    if ofile.is_file():
        print("documentation for " + str(f) + " already exists...skip")
    else:
        with ofile.open("w", encoding="utf-8") as out:
            print("writing stub documentation file for " + f)
            out.write(template.format(f, f))
