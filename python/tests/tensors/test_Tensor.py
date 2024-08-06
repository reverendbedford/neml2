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
import neml2


def test_named_ctors(tensor_options):
    batch_shape = (2, 3)
    base_shape = (3, 1, 2)
    shape = batch_shape + base_shape

    # empty
    A = neml2.Tensor.empty(base_shape, **tensor_options)
    assert A.batch.dim() == 0
    A = neml2.Tensor.empty(batch_shape, base_shape, **tensor_options)
    assert A.batch.dim() == len(batch_shape)

    # zeros
    A = neml2.Tensor.zeros(base_shape, **tensor_options)
    assert A.batch.dim() == 0
    assert torch.allclose(A.torch(), torch.zeros(base_shape, **tensor_options))
    A = neml2.Tensor.zeros(batch_shape, base_shape, **tensor_options)
    assert A.batch.dim() == len(batch_shape)
    assert torch.allclose(A.torch(), torch.zeros(shape, **tensor_options))

    # ones
    A = neml2.Tensor.ones(base_shape, **tensor_options)
    assert A.batch.dim() == 0
    assert torch.allclose(A.torch(), torch.ones(base_shape, **tensor_options))
    A = neml2.Tensor.ones(batch_shape, base_shape, **tensor_options)
    assert A.batch.dim() == len(batch_shape)
    assert torch.allclose(A.torch(), torch.ones(shape, **tensor_options))

    # full
    A = neml2.Tensor.full(base_shape, 1.1, **tensor_options)
    assert A.batch.dim() == 0
    assert torch.allclose(A.torch(), torch.full(base_shape, 1.1, **tensor_options))
    A = neml2.Tensor.full(batch_shape, base_shape, 2.3, **tensor_options)
    assert A.batch.dim() == len(batch_shape)
    assert torch.allclose(A.torch(), torch.full(shape, 2.3, **tensor_options))

    # identity
    A = neml2.Tensor.identity(5, **tensor_options)
    assert A.batch.dim() == 0
    assert torch.allclose(A.torch(), torch.eye(5, **tensor_options))
    A = neml2.Tensor.identity(batch_shape, 5, **tensor_options)
    assert A.batch.dim() == len(batch_shape)
    assert torch.allclose(
        A.torch(), torch.eye(5, **tensor_options).expand(batch_shape + (-1, -1))
    )


def test_views(tensor_options):
    batch_shape = (2, 3)
    base_shape = (3, 1, 2)
    A = neml2.Tensor.full(batch_shape, base_shape, 2.3, **tensor_options)
    A_torch = A.torch()

    assert A.device == A_torch.device
    assert A.dtype == A_torch.dtype

    A_torch[0, 2, ..., :] = 5.5

    assert torch.allclose(A.torch(), A_torch)
    assert A.device == A_torch.device
    assert A.dtype == A_torch.dtype
