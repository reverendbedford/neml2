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

# fixtures
from common import *
from test_TensorBase import sample, _A, _B, _C, base_shape
import neml2


def test_named_ctors(tensor_options):
    # identity_map
    A = neml2.Scalar.identity_map(**tensor_options)
    assert A.batch.dim() == 0


@pytest.mark.parametrize("batch_shape", [(), (2, 5, 3, 2)])
def test_binary_ops(batch_shape, sample, tensor_options):
    s = neml2.Scalar.full(batch_shape, 0.5, **tensor_options)
    s0 = s.torch()[((...,) + (None,) * sample.base.dim())]
    sample0 = sample.torch()

    # add
    result = s + sample
    correct = s0 + sample0
    assert torch.allclose(result.torch(), correct)
    result = sample + s
    correct = sample0 + s0
    assert torch.allclose(result.torch(), correct)

    # sub
    result = s - sample
    correct = s0 - sample0
    assert torch.allclose(result.torch(), correct)
    result = sample - s
    correct = sample0 - s0
    assert torch.allclose(result.torch(), correct)

    # mul
    result = s * sample
    correct = s0 * sample0
    assert torch.allclose(result.torch(), correct)
    result = sample * s
    correct = sample0 * s0
    assert torch.allclose(result.torch(), correct)

    # div
    result = s / sample
    correct = s0 / sample0
    assert torch.allclose(result.torch(), correct)
    result = sample / s
    correct = sample0 / s0
    assert torch.allclose(result.torch(), correct)

    # pow
    result = sample**s
    correct = sample0**s0
    assert torch.allclose(result.torch(), correct, equal_nan=True)
