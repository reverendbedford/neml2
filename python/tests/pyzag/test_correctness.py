"""Test we get the same answers in pyzag with parallel time as we do in NEML2 with sequential time"""

import unittest
import os.path

import torch

torch.set_default_dtype(torch.double)

from neml2.pyzag import interface
import neml2
from pyzag import nonlinear


nchunk = 10

models = ["elastic_model", "viscoplastic_model", "complex_model"]


class TestCorrectness(unittest.TestCase):
    def test_correctness(self):
        for i, model in enumerate(models):
            with self.subTest(i=i):
                self.run_model(model)

    def run_model(self, model_name):
        nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), model_name + ".i"),
            "implicit_rate",
        )
        model = interface.NEML2PyzagModel(nmodel)

        results = torch.jit.load(
            os.path.join(os.path.dirname(__file__), "result_" + model_name + ".pt")
        )

        modules = dict(results.named_modules())
        input = dict(modules["input"].named_buffers())
        output = dict(modules["output"].named_buffers())

        forces = torch.cat(
            [
                input["forces/" + n]
                for n in model.model.input_axis().subaxis("forces").variable_names()
            ],
            dim=-1,
        )
        state = torch.cat(
            [
                output["state/" + n]
                for n in model.model.input_axis().subaxis("state").variable_names()
            ],
            dim=-1,
        )

        solver = nonlinear.RecursiveNonlinearEquationSolver(
            model,
            step_generator=nonlinear.StepGenerator(nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        with torch.no_grad():
            results = nonlinear.solve(
                solver,
                state[0],
                len(forces),
                forces,
            )

        self.assertTrue(torch.allclose(state, results))
