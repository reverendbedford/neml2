#!/usr/bin/env python

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rcParams["text.usetex"] = True
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == "__main__":
    ntrial = 10
    nsample = 1000
    uts = np.empty((ntrial, nsample))
    for i in range(ntrial):
        data = pd.read_csv("samples/{}.csv".format(i + 1))
        uts[i] = data["uts"]
    uts = uts.flatten()
    cnt = np.arange(ntrial * nsample) + 1
    uts_mean = np.cumsum(uts) / cnt
    uts_std = np.sqrt(np.cumsum((uts - uts_mean) ** 2) / cnt)
    uts_kernel = sp.stats.gaussian_kde(uts)
    uts_fake = np.linspace(np.min(uts), np.max(uts), 100)
    uts_pdf = uts_kernel(uts_fake)

    fig, ax = plt.subplots()
    ax.hist(uts, bins=60, density=True)
    ax.plot(uts_fake, uts_pdf)
    ax.set_xlabel("Ultimate tensile strength (MPa)")
    fig.tight_layout()
    fig.savefig("uq.png")

    fig, ax = plt.subplots()
    ax.plot(cnt, uts_mean)
    ax.set_xlabel("Number of forward evaluations")
    ax.set_ylabel("Mean of ultimate tensile strength (MPa)")
    fig.tight_layout()
    fig.savefig("uq_mean.png")

    fig, ax = plt.subplots()
    ax.plot(cnt, uts_std)
    ax.set_xlabel("Number of forward evaluations")
    ax.set_ylabel("STD of ultimate tensile strength (MPa)")
    fig.tight_layout()
    fig.savefig("uq_std.png")
