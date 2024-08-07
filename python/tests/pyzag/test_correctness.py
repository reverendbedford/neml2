"""Test we get the same answers in pyzag with parallel time as we do in NEML2 with sequential time"""

import unittest
import os.path

import torch

torch.set_default_dtype(torch.double)

from neml2.pyzag import interface
import neml2
from pyzag import nonlinear


class CorrectnessCheck:
    def run_forward(self):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.pmodel,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        return nonlinear.solve(
            solver,
            self.initial_state.detach().clone(),
            len(self.forces),
            self.forces,
        )

    def test_correctness(self):
        pyzag_result = self.run_forward()
        print(pyzag_result.shape)


class TestElasticModel(unittest.TestCase, CorrectnessCheck):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "elastic_model.i"), "implicit_rate"
        )
        self.pmodel = interface.NEML2PyzagModel(self.nmodel)

        self.nbatch = 20
        self.ntime = 100

        end_time = torch.logspace(-1, -5, self.nbatch)
        time = torch.stack(
            [torch.linspace(0, et, self.ntime) for et in end_time]
        ).T.unsqueeze(-1)
        strain = (
            torch.stack(
                [
                    torch.linspace(0, 0.1, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                ]
            )
            .T[:, None]
            .expand(-1, self.nbatch, -1)
        )

        self.initial_state = torch.zeros((self.nbatch, 6))
        self.forces = self.pmodel.collect_forces({"t": time, "E": strain})

        self.nchunk = 10


class TestViscoplasticModel(unittest.TestCase, CorrectnessCheck):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "viscoplastic_model.i"),
            "implicit_rate",
        )
        self.pmodel = interface.NEML2PyzagModel(self.nmodel)

        self.nbatch = 20
        self.ntime = 100

        end_time = torch.logspace(-1, -5, self.nbatch)
        time = torch.stack(
            [torch.linspace(0, et, self.ntime) for et in end_time]
        ).T.unsqueeze(-1)
        strain = (
            torch.stack(
                [
                    torch.linspace(0, 0.1, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                ]
            )
            .T[:, None]
            .expand(-1, self.nbatch, -1)
        )

        self.initial_state = torch.zeros((self.nbatch, 7))
        self.forces = self.pmodel.collect_forces({"t": time, "E": strain})

        self.nchunk = 10


class TestComplexModel(unittest.TestCase, CorrectnessCheck):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "complex_model.i"),
            "model",
        )
        self.pmodel = interface.NEML2PyzagModel(
            self.nmodel, exclude_parameters=["yield_zero.sy"]
        )

        self.nbatch = 20
        self.ntime = 100

        end_time = torch.logspace(-1, -5, self.nbatch)
        time = torch.stack(
            [torch.linspace(0, et, self.ntime) for et in end_time]
        ).T.unsqueeze(-1)
        conditions = (
            torch.stack(
                [
                    torch.linspace(0, 0.1, self.ntime),
                    torch.linspace(0, -50, self.ntime),
                    torch.linspace(0, -0.025, self.ntime),
                    torch.linspace(0, 0.15, self.ntime),
                    torch.linspace(0, 75.0, self.ntime),
                    torch.linspace(0, 0.05, self.ntime),
                ]
            )
            .T[:, None]
            .expand(-1, self.nbatch, -1)
        )

        control = torch.zeros((self.ntime, self.nbatch, 6))
        control[..., 1] = 1.0
        control[..., 4] = 1.0

        temperatures = torch.stack(
            [
                torch.linspace(T1, T2, self.ntime)
                for T1, T2 in zip(
                    torch.linspace(300, 500, self.nbatch),
                    torch.linspace(600, 1200, self.nbatch),
                )
            ]
        ).T.unsqueeze(-1)

        self.initial_state = torch.zeros((self.nbatch, 8))
        self.forces = self.pmodel.collect_forces(
            {
                "t": time,
                "control": control,
                "fixed_values": conditions,
                "T": temperatures,
            }
        )

        self.nchunk = 10
