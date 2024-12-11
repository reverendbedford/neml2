# Copyright 2024, UChicago Argonne, LLC
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

"""Test adjoint correctness"""

import pytest
from pathlib import Path
import torch
import neml2
from neml2.tensors import Scalar, SR2
from pyzag import nonlinear


class DerivativeCheck:
    def adjoint_grads(self):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        solver.zero_grad()
        res = nonlinear.solve_adjoint(solver, self.initial_state, self.nstep, self.forces)
        val = torch.norm(res)
        val.backward()
        return {n: p.grad for n, p in solver.named_parameters()}

    def fd_grads(self, eps=1.0e-6):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        res = {}
        with torch.no_grad():
            val0 = torch.norm(nonlinear.solve(solver, self.initial_state, self.nstep, self.forces))
            for n, p in solver.named_parameters():
                p0 = p.clone()
                dx = torch.abs(p0) * eps
                p.data = p0 + dx
                val1 = torch.norm(
                    nonlinear.solve(solver, self.initial_state, self.nstep, self.forces)
                )
                res[n] = (val1 - val0) / dx
                p.data = p0
        return res

    def test_adjoint_vs_fd(self):
        grads_adjoint = self.adjoint_grads()
        grads_fd = self.fd_grads()
        assert grads_adjoint.keys() == grads_fd.keys()
        for n in grads_adjoint.keys():
            assert torch.allclose(grads_adjoint[n], grads_fd[n], atol=self.atol, rtol=self.rtol)


class TestElasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = neml2.load_model(pwd / "elastic_model.i", "implicit_rate")
        self.model = neml2.pyzag.NEML2PyzagModel(nmodel)

        self.nbatch = 20
        self.nstep = 100

        # Prescribed time
        start_time = Scalar.full(0.0)
        end_time = Scalar(torch.logspace(-1, -5, self.nbatch))
        time = Scalar.linspace(start_time, end_time, self.nstep)

        # Prescribed strain
        start_strain = SR2.full(0.0)
        end_strain = SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0)
        strain = SR2.linspace(start_strain, end_strain, self.nstep).batch.unsqueeze(-1)

        # Prescribed forces
        self.forces = self.model.forces_asm.assemble({"forces/t": time, "forces/E": strain}).torch()

        # Initial state
        self.initial_state = torch.full((self.nbatch, self.model.nstate), 0.0)

        # Parallel time integration
        self.nchunk = 10

        # Tolerances
        self.atol = 1e-8
        self.rtol = 1e-5


class TestViscoplasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = neml2.load_model(pwd / "viscoplastic_model.i", "implicit_rate")
        self.model = neml2.pyzag.NEML2PyzagModel(nmodel)

        self.nbatch = 20
        self.nstep = 100

        # Prescribed time
        start_time = Scalar.full(0.0)
        end_time = Scalar(torch.logspace(-1, -5, self.nbatch))
        time = Scalar.linspace(start_time, end_time, self.nstep)

        # Prescribed strain
        start_strain = SR2.full(0.0)
        end_strain = SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0)
        strain = SR2.linspace(start_strain, end_strain, self.nstep).batch.unsqueeze(-1)

        # Prescribed forces
        self.forces = self.model.forces_asm.assemble({"forces/t": time, "forces/E": strain}).torch()

        # Initial state
        self.initial_state = torch.full((self.nbatch, self.model.nstate), 0.0)

        # Parallel time integration
        self.nchunk = 10

        # Tolerances
        self.atol = 1e-8
        self.rtol = 1e-5


# class TestComplexModel(unittest.TestCase, DerivativeCheck):
#     def setUp(self):
#         self.nmodel = neml2.load_model(
#             os.path.join(os.path.dirname(__file__), "complex_model.i"),
#             "implicit_rate",
#         )
#         self.pmodel = neml2.pyzag.NEML2PyzagModel(
#             self.nmodel, exclude_parameters=["yield_zero.sy", "mu.X", "mu.Y"]
#         )
#         self.nbatch = 20
#         self.ntime = 100

#         end_time = torch.logspace(-1, -5, self.nbatch)
#         time = torch.stack([torch.linspace(0, et, self.ntime) for et in end_time]).T.unsqueeze(-1)
#         conditions = (
#             torch.stack(
#                 [
#                     torch.linspace(0, 0.1, self.ntime),
#                     torch.linspace(0, -50, self.ntime),
#                     torch.linspace(0, -0.025, self.ntime),
#                     torch.linspace(0, 0.15, self.ntime),
#                     torch.linspace(0, 75.0, self.ntime),
#                     torch.linspace(0, 0.05, self.ntime),
#                 ]
#             )
#             .T[:, None]
#             .expand(-1, self.nbatch, -1)
#         )

#         control = torch.zeros((self.ntime, self.nbatch, 6))
#         control[..., 1] = 1.0
#         control[..., 4] = 1.0

#         temperatures = torch.stack(
#             [
#                 torch.linspace(T1, T2, self.ntime)
#                 for T1, T2 in zip(
#                     torch.linspace(300, 500, self.nbatch),
#                     torch.linspace(600, 1200, self.nbatch),
#                 )
#             ]
#         ).T.unsqueeze(-1)

#         self.initial_state = torch.zeros((self.nbatch, 8))
#         self.forces = self.pmodel.collect_forces(
#             {
#                 "t": time,
#                 "control": control,
#                 "fixed_values": conditions,
#                 "T": temperatures,
#             }
#         )

#         self.nchunk = 10

#         self.rtol = 1.0e-4
