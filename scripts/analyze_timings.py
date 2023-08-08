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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rcParams["text.usetex"] = True
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

data = pd.read_csv(sys.argv[1]).filter(["device", "nbatch", "mean", "std"])

devices = data["device"].unique()

fig, ax = plt.subplots(figsize=(6, 4))

for device in devices:
    timings = data.loc[data["device"] == device]
    timings = timings.sort_values("nbatch")
    ax.errorbar(
        timings["nbatch"],
        timings["mean"],
        marker="o",
        yerr=timings["std"],
        label=device,
    )

# Draw an arrow between cpu and cuda to indicate speedup
max_nbatch = np.max(data["nbatch"])
timing_cpu = data.loc[(data["nbatch"] == max_nbatch) & (data["device"] == "cpu")][
    "mean"
].iloc[0]
timing_cuda = data.loc[(data["nbatch"] == max_nbatch) & (data["device"] == "cuda")][
    "mean"
].iloc[0]
speedup = timing_cpu / timing_cuda

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of batches")
ax.set_ylabel(r"Time per iteration ($s$)")
ax.legend()
fig.suptitle("Speed up = {:.0f} @ batch size = {:d}".format(speedup, max_nbatch))
fig.tight_layout()
fig.savefig(sys.argv[2])
