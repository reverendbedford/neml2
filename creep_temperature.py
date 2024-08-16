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
import math
import torch
import neml2
import numpy as np
from neml2.tensors import LabeledVector, Tensor
from matplotlib import pyplot as plt
from matplotlib import cm, colors

torch.set_default_dtype(torch.float64)

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


def integrate(
    model,
    input_forces,
    input_old_forces,
    input_state,
    input_old_state,
    output_state,
    forces,
    initial_state,
):
    # Number of steps to integrate
    shapes = [f.shape[:-1] for _, f in forces.items()]
    assert shapes.count(shapes[0]) == len(forces)
    shape = shapes[0]
    nstep = shape[0]
    bshape = shape[1:]

    # ndof
    ndof_input = model.input_axis().storage_size()
    ndof_output = model.output_axis().storage_size()

    # Allocate storage for inputs and outputs
    inputs = torch.zeros(*shape, ndof_input)
    outputs = torch.zeros(*shape, ndof_output)

    # Fill input forces
    for force_name, force in forces.items():
        inputs[..., input_forces[force_name]] = force

    # Fill initial state
    for state_name, state in initial_state.items():
        outputs[0, ..., output_state[state_name]] = state

    # Step through the prescribed forces to integrate the model
    for i in range(1, nstep):
        # Set old forces to be forces from the previous step
        for force_name, index in input_old_forces.items():
            inputs[i, ..., index] = inputs[i - 1, ..., input_forces[force_name]]
        # Set old state to be state from the previous step
        for state_name, index in input_old_state.items():
            inputs[i, ..., index] = inputs[i - 1, ..., input_state[state_name]]
        # Use old state as the initial guess for the current state
        for state_name, index in input_state.items():
            inputs[i, ..., index] = inputs[i, ..., input_old_state[state_name]]

        # Evaluate the model
        x = inputs[i]
        y = model.value(x)

        # Collect output
        outputs[i] = y

    return outputs


if __name__ == "__main__":
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "creep.i", "model")

    # Batch shape is (ntemperature, nsmax, nmaterial)
    smax = [50]
    temperature = [
        300,
        310,
        320,
        340,
        380,
        460,
        620,
        940,
    ]
    ntemperature = len(temperature)
    nsmax = len(smax)
    nmaterial = 1
    B = (ntemperature, nsmax, nmaterial)

    # Initialize the model with the correct batch shape
    # Derivative order is 0 since we don't care about dy/dx, we only need parameter gradient dy/dp
    model.reinit(batch_shape=B, deriv_order=0)

    ########################################
    # input axis:
    #              forces/S: 0:6:1
    #              forces/T: 6:7:1
    #              forces/t: 7:8:1
    #          old_forces/t: 8:9:1
    # old_state/internal/Ep: 9:15:1
    # old_state/internal/ep: 15:16:1
    #     state/internal/Ep: 16:22:1
    #     state/internal/ep: 22:23:1
    ########################################
    # output axis:
    #           state/E: 0:6:1
    # state/internal/Ep: 6:12:1
    # state/internal/ep: 12:13:1
    ########################################
    # parameters:
    #   elastic_strain.E
    #   elastic_strain.nu
    #   flow_rate.eta
    #   flow_rate.n
    #   isoharden.R
    #   isoharden.d
    #   yield.sy
    ########################################

    # First, let's create a dictionary maintaining variable indices
    #
    # I'll have to do it manually here. You'll be able to do this automatically
    # using pyzag (which will be publicly released later this year)
    input_forces = {"S": slice(0, 6), "T": slice(6, 7), "t": slice(7, 8)}
    input_old_forces = {"t": slice(8, 9)}
    input_state = {"Ep": slice(16, 22), "ep": slice(22, 23)}
    input_old_state = {"Ep": slice(9, 15), "ep": slice(15, 16)}
    output_state = {"E": slice(0, 6), "Ep": slice(6, 12), "ep": slice(12, 13)}

    # To integrate the model, we need three things
    #   1. Prescribed forces
    #   2. Prescribed temperatures
    #   3. Initial state

    # For a uniaxial stress test, we first ramp up the stress
    # over a short period of time, then hold it constant
    tramp = 360
    nramp = 10
    nload = 60
    ntime = nramp + nload
    prescribed_t = torch.cat(
        [torch.linspace(0, tramp, nramp), torch.logspace(1, 6, nload) + tramp]
    )[..., None, None, None].expand(ntime, *B)[..., None]
    prescribed_S = torch.zeros(ntime, *B, 6)
    for i, s in enumerate(smax):
        prescribed_Sx = torch.cat(
            [torch.linspace(0, s, nramp), torch.full((nload,), s)]
        )[..., None, None]
        prescribed_S[:, :, i, :, 0] = prescribed_Sx
    prescribed_T = torch.empty((70, *B, 1))
    for i, T in enumerate(temperature):
        prescribed_T[:, i] = T

    # Initial plastic strains are zero
    initial_Ep = torch.zeros(6)
    initial_ep = torch.zeros(1)

    # Now we have everything needed to integrate the model
    outputs = integrate(
        model,
        input_forces,
        input_old_forces,
        input_state,
        input_old_state,
        output_state,
        {"S": prescribed_S, "T": prescribed_T, "t": prescribed_t},
        {"Ep": initial_Ep, "ep": initial_ep},
    )

    # We can then plot strain and strain rate
    strain = outputs[..., 0].squeeze()
    ep = outputs[..., 12].squeeze()
    times = prescribed_t.squeeze()
    strainrate = torch.diff(strain, dim=0) / torch.diff(times, dim=0)
    t0 = times[1, 0].item()
    norm = colors.Normalize(vmin=np.min(temperature), vmax=np.max(temperature))
    sm = cm.ScalarMappable(norm=norm, cmap="rainbow")

    fig, ax = plt.subplots()
    for i, T in enumerate(temperature):
        ax.plot(
            times[1:, i] / 3600, strainrate[:, i] * 100 * 3600, "-", color=sm.to_rgba(T)
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Time (hr)",
        ylabel="Strain rate (\\%/hr)",
    )
    ax.set_xlim(tramp * 1.1 / 3600)
    ax.set_ylim(4e-6, 2e2)
    fig.tight_layout()
    fig.colorbar(sm, ax=ax, label="Temperature (K)")
    fig.savefig("strainrate2.png")
    fig.savefig("strainrate2.pdf")

    fig, ax = plt.subplots()
    for i, T in enumerate(temperature):
        ax.plot(times[1:, i] / 3600, ep[1:, i], "-", color=sm.to_rgba(T))
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Time (hr)",
        ylabel="Equivalent plastic strain",
    )
    ax.set_xlim(tramp * 1.1 / 3600)
    ax.set_ylim(1e-10, 1e3)
    fig.tight_layout()
    fig.colorbar(sm, ax=ax, label="Temperature (K)")
    fig.savefig("eqpstrain2.png")
    fig.savefig("eqpstrain2.pdf")
