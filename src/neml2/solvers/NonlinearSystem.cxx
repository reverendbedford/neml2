// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
BatchTensor
NonlinearSystem::residual(const BatchTensor & x) const
{
  neml_assert_dbg(x.base_dim() == 1, "The residual must be a logical vector, i.e. base_dim() == 1");
  auto r = BatchTensor::empty_like(x);
  assemble(x, &r);
  return r;
}

BatchTensor
NonlinearSystem::Jacobian(const BatchTensor & x) const
{
  neml_assert_dbg(x.base_dim() == 1, "The residual must be a logical vector, i.e. base_dim() == 1");
  auto n = x.base_sizes()[0];
  auto J = BatchTensor::zeros(x.batch_sizes(), {n, n}, x.options());
  assemble(x, nullptr, &J);
  return J;
}

std::tuple<BatchTensor, BatchTensor>
NonlinearSystem::residual_and_Jacobian(const BatchTensor & x) const
{
  neml_assert_dbg(x.base_dim() == 1, "The residual must be a logical vector, i.e. base_dim() == 1");
  auto n = x.base_sizes()[0];
  auto r = BatchTensor::empty_like(x);
  auto J = BatchTensor::zeros(x.batch_sizes(), {n, n}, x.options());
  assemble(x, &r, &J);
  return {r, J};
}
} // namespace neml2
