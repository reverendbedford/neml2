#!/usr/bin/env python

from pathlib import Path
import torch
import neml2
from scipy.stats import beta, gamma, loguniform, uniform, norm
import numpy as np
from tqdm import tqdm
import pandas as pd

torch.set_default_dtype(torch.float64)


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
    for i in tqdm(range(1, nstep)):
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
    id = 10
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)

    print(model.input_axis())
    print(model.output_axis())
    print(model.named_parameters().keys())

    nsample = 1000
    B = (nsample,)

    # Initialize the model with the correct batch shape
    # Derivative order is 0 since we don't care about dy/dx, we only need parameter gradient dy/dp
    model.reinit(batch_shape=B, deriv_order=0)

    ########################################
    # input axis:
    #              forces/E: 0:6:1
    #              forces/t: 6:7:1
    #          old_forces/t: 7:8:1
    # old_state/internal/Ep: 8:14:1
    # old_state/internal/X1: 14:20:1
    # old_state/internal/X2: 20:26:1
    # old_state/internal/ep: 26:27:1
    #     state/internal/Ep: 27:33:1
    #     state/internal/X1: 33:39:1
    #     state/internal/X2: 39:45:1
    #     state/internal/ep: 45:46:1
    ########################################
    # output axis:
    #           state/S: 0:6:1
    # state/internal/Ep: 6:12:1
    # state/internal/X1: 12:18:1
    # state/internal/X2: 18:24:1
    # state/internal/ep: 24:25:1
    #  state/internal/s: 25:26:1
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
        "t": slice(7, 8),
    }
    input_state = {
        "Ep": slice(27, 33),
        "X1": slice(33, 39),
        "X2": slice(39, 45),
        "ep": slice(45, 46),
    }
    input_old_state = {
        "Ep": slice(8, 14),
        "X1": slice(14, 20),
        "X2": slice(20, 26),
        "ep": slice(26, 27),
    }
    output_state = {
        "Ep": slice(6, 12),
        "X1": slice(12, 18),
        "X2": slice(18, 24),
        "ep": slice(24, 25),
    }

    # Parameter distributions
    # 'X1rate.A'
    # 'X1rate.C'
    # 'X1rate.a'
    # 'X1rate.g'
    # 'X2rate.A'
    # 'X2rate.C'
    # 'X2rate.a'
    # 'X2rate.g'
    # 'elasticity.E'
    # 'elasticity.nu'
    # 'flow_rate.eta'
    # 'flow_rate.n'
    # 'isoharden1.R'
    # 'isoharden1.d'
    # 'isoharden2.R'
    # 'isoharden2.d'
    # 'yield.sy'
    param_dist = {
        "X1rate.A": loguniform(1e-10, 1e-5),
        "X1rate.C": gamma(1e3, scale=0.1),
        "X1rate.a": uniform(2, 5),
        "X1rate.g": norm(1, scale=0.1),
        "X2rate.A": loguniform(1e-9, 1e-6),
        "X2rate.C": gamma(5e3, scale=0.02),
        "X2rate.a": uniform(3, 6),
        "X2rate.g": norm(2, scale=0.2),
        "elasticity.E": gamma(1e4, scale=0.5),
        "elasticity.nu": beta(2, 1.5, scale=0.5),
        "flow_rate.eta": norm(100, scale=3),
        "flow_rate.n": uniform(3, 5),
        "isoharden1.R": norm(100, scale=3),
        "isoharden1.d": norm(150, scale=2),
        "isoharden2.R": norm(-80, scale=2),
        "isoharden2.d": norm(80, scale=3),
        "yield.sy": gamma(5e3, scale=0.01),
    }

    # To integrate the model, we need two things
    #   1. Prescribed forces
    #   2. Initial state

    # For uniaxial tension, we ramp up the strain over a short period of time
    tend = 600
    ntime = 300
    emax = 0.2
    prescribed_t = torch.linspace(0, tend, ntime)[..., None, None].expand(-1, *B, 1)
    prescribed_E = torch.zeros(ntime, *B, 6)
    prescribed_E[..., 0] = torch.linspace(0, emax, ntime)[..., None].expand(-1, *B)
    prescribed_E[..., 1] = -0.5 * prescribed_E[..., 0]
    prescribed_E[..., 2] = -0.5 * prescribed_E[..., 0]

    # Initial condition
    initial_Ep = torch.zeros(6)
    initial_X1 = torch.zeros(6)
    initial_X2 = torch.zeros(6)
    initial_ep = torch.zeros(1)

    for param, dist in param_dist.items():
        pval = dist.rvs(size=nsample)
        print("{}: {:.3E}+-{:.3E}".format(param, np.mean(pval), np.std(pval)))
        model.set_parameter(param, neml2.Tensor(torch.tensor(pval), 1))
    outputs = integrate(
        model,
        input_forces,
        input_old_forces,
        input_state,
        input_old_state,
        output_state,
        {"E": prescribed_E, "t": prescribed_t},
        {"Ep": initial_Ep, "ep": initial_ep, "X1": initial_X1, "X2": initial_X2},
    )
    stress = outputs[..., 25]
    uts, _ = torch.max(stress, dim=0)
    data = {"uts": uts.numpy()}
    df = pd.DataFrame(data)
    df.to_csv("samples/{}.csv".format(id), index=False)
