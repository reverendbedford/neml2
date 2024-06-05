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
from neml2.tensors import BatchTensor, SR2, LabeledVector


def test_named_parameters():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_ParameterStore.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    E = model.named_parameters()["E"]
    nu = model.named_parameters()["nu"]

    # Parameters should have the correct value
    assert torch.allclose(E.tensor().tensor(), torch.tensor(100.0, dtype=torch.float64))
    assert torch.allclose(nu.tensor().tensor(), torch.tensor(0.3, dtype=torch.float64))


def test_parameter_derivative():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_ParameterStore.i", "model")

    # Strain of batch shape (5,2)
    e = SR2(torch.tensor([0.1, 0.2, 0.05, 0, 0, 0]).expand(5, 2, 6))

    # The input vector only contains strain
    x = LabeledVector(BatchTensor(e), [model.input_axis()])

    # Setup variable views
    model.reinit(x.tensor().batch.shape)

    # This model has two parameters
    E = model.named_parameters()["E"]
    nu = model.named_parameters()["nu"]

    # Model parameters do not require grad by default
    assert not E.requires_grad
    assert not nu.requires_grad

    # Set parameters to requires_grad=True
    E.requires_grad_(True)
    nu.requires_grad_(True)
    assert E.requires_grad
    assert nu.requires_grad

    # Forward
    y = model.value(x)
    assert y.tensor().requires_grad

    # dy/dp * x
    y.tensor().tensor().backward(gradient=x.tensor().tensor())
    assert torch.allclose(E.grad, torch.tensor(1.1105769561746945, dtype=torch.float64))
    assert torch.allclose(
        nu.grad, torch.tensor(503.51332861533353, dtype=torch.float64)
    )
