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

import pytest
from pathlib import Path
import torch
import neml2
from neml2.tensors import Tensor, LabeledVector, LabeledMatrix, TensorType


def test_get_model():
    pwd = Path(__file__).parent
    neml2.reload_input(pwd / "test_Model.i")

    model1 = neml2.get_model("model")
    assert model1.is_AD_enabled

    model2 = neml2.get_model("model", enable_AD=False)
    assert not model2.is_AD_enabled


def test_diagnose():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_Model_diagnose.i", "model")
    diagnoses = neml2.diagnose(model)
    assert (
        "This model is part of a nonlinear system. At least one of the input variables is solve-dependent, so all output variables MUST be solve-dependent"
        in diagnoses
    )


def test_input_type():
    pwd = Path(__file__).parent
    neml2.reload_input(pwd / "test_training.i")
    model = neml2.get_model("model")
    assert model.input_type("forces/E") == TensorType.SR2
    assert model.input_type("forces/t") == TensorType.Scalar
    assert model.input_type("old_forces/E") == TensorType.SR2
    assert model.input_type("old_forces/t") == TensorType.Scalar
    assert model.input_type("state/S") == TensorType.SR2
    assert model.input_type("old_state/S") == TensorType.SR2


def test_output_type():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_training.i", "model")
    assert model.output_type("state/S") == TensorType.SR2


def test_dependency():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_Model.i", "model")
    model.reinit()
    submodels = model.named_submodels()

    assert submodels["foo"].dependency()["forces/t"].name == "model"
    assert submodels["foo"].dependency()["old_forces/t"].name == "model"
    assert submodels["foo"].dependency()["old_state/foo"].name == "model"
    assert submodels["foo"].dependency()["state/foo"].name == "model"
    assert submodels["foo"].dependency()["state/foo_rate"].name == "model"

    assert submodels["bar"].dependency()["forces/t"].name == "model"
    assert submodels["bar"].dependency()["old_forces/t"].name == "model"
    assert submodels["bar"].dependency()["old_state/bar"].name == "model"
    assert submodels["bar"].dependency()["state/bar"].name == "model"
    assert submodels["bar"].dependency()["state/bar_rate"].name == "model"

    assert submodels["baz"].dependency()["residual/foo"].name == "foo"
    assert submodels["baz"].dependency()["residual/bar"].name == "bar"


def test_value():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_Model.i", "model")

    a = torch.linspace(0, 1, 8).expand(5, 2, 8)
    x = Tensor(a, 2)
    model.reinit(x.batch.shape)
    x = LabeledVector(x, [model.input_axis()])

    y_correct = torch.tensor(0.9591836144729537, dtype=torch.float64)

    # Evaluate the model using a torch.Tensor
    y = model.value(x.torch())
    assert isinstance(y, torch.Tensor)
    assert y.shape == (5, 2, 1)
    assert torch.allclose(y, y_correct)

    # Evaluate the model using a neml2.Tensor
    y = model.value(x.tensor())
    assert isinstance(y, Tensor)
    assert y.batch.shape == (5, 2)
    assert y.base.shape == (1,)
    assert torch.allclose(y.torch(), y_correct)

    # Evaluate the model using a neml2.LabeledVector
    y = model.value(x)
    assert isinstance(y, LabeledVector)
    assert y.batch.shape == (5, 2)
    assert y.base.shape == (1,)
    assert torch.allclose(y.torch(), y_correct)


def test_value_and_dvalue():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_Model.i", "model")

    a = torch.linspace(0, 1, 8).expand(5, 2, 8)
    x = Tensor(a, 2)
    model.reinit(x.batch.shape, 1)
    x = LabeledVector(x, [model.input_axis()])

    y_correct = torch.tensor(0.9591836144729537, dtype=torch.float64)
    dy_dx_correct = torch.tensor(
        [
            -1.7142857313156128,
            1.7142857313156128,
            -1.0,
            -1.0,
            1.0,
            0.1428571492433548,
            1.0,
            0.1428571492433548,
        ],
        dtype=torch.float64,
    )

    # Evaluate the model using a torch.Tensor
    y, dy_dx = model.value_and_dvalue(x.torch())
    assert isinstance(y, torch.Tensor)
    assert isinstance(dy_dx, torch.Tensor)
    assert y.shape == (5, 2, 1)
    assert dy_dx.shape == (5, 2, 1, 8)
    assert torch.allclose(y, y_correct)
    assert torch.allclose(dy_dx, dy_dx_correct)

    # Evaluate the model using a neml2.Tensor
    y, dy_dx = model.value_and_dvalue(x.tensor())
    assert isinstance(y, Tensor)
    assert isinstance(dy_dx, Tensor)
    assert y.batch.shape == (5, 2)
    assert y.base.shape == (1,)
    assert dy_dx.batch.shape == (5, 2)
    assert dy_dx.base.shape == (1, 8)
    assert torch.allclose(y.torch(), y_correct)
    assert torch.allclose(dy_dx.torch(), dy_dx_correct)

    # Evaluate the model using a neml2.LabeledVector
    y, dy_dx = model.value_and_dvalue(x)
    assert isinstance(y, LabeledVector)
    assert isinstance(dy_dx, LabeledMatrix)
    assert y.batch.shape == (5, 2)
    assert y.base.shape == (1,)
    assert dy_dx.batch.shape == (5, 2)
    assert dy_dx.base.shape == (1, 8)
    assert torch.allclose(y.torch(), y_correct)
    assert torch.allclose(dy_dx.torch(), dy_dx_correct)
