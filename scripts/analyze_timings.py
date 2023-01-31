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


import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1]).filter(
    ["test_case", "benchmark", "nbatch", "mean", "low_mean", "high_mean"]
)

test_cases = set(data["test_case"])

for test_case in test_cases:
    print("test case: " + test_case)
    benchmarks = set(data.loc[(data["test_case"] == test_case)]["benchmark"])
    for benchmark in benchmarks:
        print("     benchmark: " + benchmark)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(test_case + "\n" + benchmark)
        select = data.loc[
            (data["test_case"] == test_case) & (data["benchmark"] == benchmark)
        ]
        ax.plot(select["nbatch"], select["mean"], "o-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of batches")
        ax.set_ylabel("Time per iteration [s]")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        path = Path(sys.argv[2]) / test_case
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / (benchmark + ".png"), dpi=300)
        plt.close()
