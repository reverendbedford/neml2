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


import pytest
from pathlib import Path
import neml2
import torch


def test_tensor_properties():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    v = neml2.LabeledVector.zeros(
        (5, 2), [model.input_axis()], dtype=torch.float32, device=torch.device("cpu")
    )
    assert len(v.axes) == 1
    assert v.axes[0] == model.input_axis()
    assert v.dim() == 3
    assert v.shape == (5, 2, 8)
    assert v.batched()
    assert v.dtype == torch.float32
    assert v.device == torch.device("cpu")
    assert not v.requires_grad


def test_batchview():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    v = neml2.LabeledVector.zeros(
        (5, 2), [model.input_axis()], device=torch.device("cpu")
    )
    assert v.batch.dim() == 2
    assert v.batch.shape == (5, 2)

    v.batch[:3, ...] = torch.ones(3, 2, 8)
    vt = v.torch()
    assert torch.allclose(vt[:3, :, :], torch.ones(3, 2, 8))
    assert torch.allclose(vt[3:, :, :], torch.zeros(2, 2, 8))


def test_baseview():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    v = neml2.LabeledVector.zeros(
        (5, 2), [model.input_axis()], device=torch.device("cpu")
    )
    assert v.base.dim() == 1
    assert v.base.shape == (8,)

    v.base["forces/t"] = neml2.Scalar.full(1.1)
    v.base["state/foo"] = neml2.Scalar.full(5.5)
    vt = v.torch()
    assert torch.allclose(vt[:, :, 0], torch.tensor(1.1))
    assert torch.allclose(vt[:, :, 6], torch.tensor(5.5))
