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
        print("step", i)
        # Set old forces to be forces from the previous step
        for force_name, index in input_old_forces.items():
            inputs[i, ..., index] = inputs[i - 1, ..., input_forces[force_name]]
        # Set old state to be state from the previous step
        for state_name, index in input_old_state.items():
            inputs[i, ..., index] = outputs[i - 1, ..., output_state[state_name]]
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
    model = neml2.load_model(pwd / "chaboche.i", "model")
    print(model.input_axis())
    print(model.output_axis())
    print(model.named_parameters().keys())

    B = (10,)

    # Initialize the model with the correct batch shape
    # Derivative order is 0 since we don't care about dy/dx, we only need parameter gradient dy/dp
    model.reinit(batch_shape=B, deriv_order=0)

    ########################################
    # input axis:
    #              forces/E: 0:6:1
    #              forces/t: 6:7:1
    #          old_forces/E: 7:13:1
    #          old_forces/t: 13:14:1
    #           old_state/S: 14:20:1
    # old_state/internal/X1: 20:26:1
    # old_state/internal/X2: 26:32:1
    # old_state/internal/ep: 32:33:1
    #               state/S: 33:39:1
    #     state/internal/X1: 39:45:1
    #     state/internal/X2: 45:51:1
    #     state/internal/ep: 51:52:1
    ########################################
    # output axis:
    #           state/S: 0:6:1
    # state/internal/X1: 6:12:1
    # state/internal/X2: 12:18:1
    # state/internal/ep: 18:19:1
    ########################################

    # First, let's create a dictionary maintaining variable indices
    #
    # I'll have to do it manually here. You'll be able to do this automatically
    # using pyzag (which will be publicly released later this year)
    input_forces = {
        "E": slice(0, 6),
        "t": slice(6, 7),
    }
    input_old_forces = {
        "E": slice(7, 13),
        "t": slice(13, 14),
    }
    input_state = {
        "S": slice(33, 39),
        "X1": slice(39, 45),
        "X2": slice(45, 51),
        "ep": slice(51, 52),
    }
    input_old_state = {
        "S": slice(14, 20),
        "X1": slice(20, 26),
        "X2": slice(26, 32),
        "ep": slice(32, 33),
    }
    output_state = {
        "S": slice(0, 6),
        "X1": slice(6, 12),
        "X2": slice(12, 18),
        "ep": slice(18, 19),
    }

    # To integrate the model, we need two things
    #   1. Prescribed forces
    #   2. Initial state

    # For a uniaxial tension test, we ramp up the strain
    # over a short period of time
    tend = 600
    ntime = 300
    Emax = 0.2
    prescribed_t = (
        torch.linspace(0, tend, ntime).unsqueeze(-1).unsqueeze(-1).expand(-1, *B, -1)
    )
    prescribed_E = torch.zeros(ntime, *B, 6)
    prescribed_E[..., 0] = torch.linspace(0, Emax, ntime).unsqueeze(-1)
    prescribed_E[..., 1] = torch.linspace(0, -Emax / 2, ntime).unsqueeze(-1)
    prescribed_E[..., 2] = torch.linspace(0, -Emax / 2, ntime).unsqueeze(-1)

    # Initial condition
    initial_S = torch.zeros(6)
    initial_X1 = torch.zeros(6)
    initial_X2 = torch.zeros(6)
    initial_ep = torch.zeros(1)

    # Now we have everything needed to integrate the model
    outputs = integrate(
        model,
        input_forces,
        input_old_forces,
        input_state,
        input_old_state,
        output_state,
        {"E": prescribed_E, "t": prescribed_t},
        {"S": initial_S, "X1": initial_X1, "X2": initial_X2, "ep": initial_ep},
    )

    strain = prescribed_E[..., 0]
    stress = outputs[..., 0]
    fig, ax = plt.subplots()
    ax.plot(strain, stress)
    fig.savefig("uq.png")
