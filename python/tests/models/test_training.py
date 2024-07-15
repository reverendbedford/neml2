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
from neml2.tensors import LabeledVector, Tensor


def test_parameter_gradient():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_training.i", "model")

    # Initialize the model with the correct batch shape
    B = (2, 5)
    model.reinit(batch_shape=B, deriv_order=1)

    # Define the input
    ndof = 26
    x = torch.linspace(0, 0.2, ndof).expand(*B, -1)
    x = LabeledVector(Tensor(x, len(B)), [model.input_axis()])

    # Say I want to get the parameter gradient on the flow viscosity
    p = model.named_parameters()["flow_rate.eta"]
    p.requires_grad_(True)

    # Evaluate the model
    y = model.value(x)

    # Evaluate the loss function
    f = torch.norm(y.torch())

    # Get the parameter gradient
    f.backward()
    assert math.isclose(p.grad.item(), 0.023917, rel_tol=1e-6, abs_tol=1e-6)
