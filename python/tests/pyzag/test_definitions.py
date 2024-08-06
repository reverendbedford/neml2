"""Test basic setup of NEML2 models"""

import unittest
import os.path

import neml2
from neml2.pyzag import interface

import torch

torch.set_default_dtype(torch.double)


class TestCorrectlyDefinedModel(unittest.TestCase):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "correct_model.i"), "implicit_rate"
        )
        self.pmodel = interface.NEML2PyzagModel(
            self.nmodel, exclude_parameters=["elasticity.nu"]
        )

    def test_parameter_names(self):
        """Check that names and parameter values are the same"""
        self.assertEqual(
            set([n for n, _ in self.pmodel.named_parameters()]),
            set(
                [
                    "elasticity_E",
                    "flow_rate_eta",
                    "flow_rate_n",
                    "isoharden_K",
                    "yield_sy",
                ]
            ),
        )

        for name, val in self.pmodel.named_parameters():
            self.assertTrue(
                torch.allclose(
                    val,
                    self.pmodel.model.named_parameters()[
                        self.pmodel.parameter_name_map[name]
                    ].torch(),
                )
            )
