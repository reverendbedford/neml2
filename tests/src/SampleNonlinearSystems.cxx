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

#include "SampleNonlinearSystems.h"
#include "neml2/tensors/Scalar.h"

using namespace torch::indexing;
using namespace neml2;

TestNonlinearSystem::TestNonlinearSystem(const OptionSet & options)
  : NonlinearSystem(options)
{
}

void
TestNonlinearSystem::reinit(const BatchTensor & x)
{
  neml_assert_dbg(x.base_dim() == 1, "Trial solution must be one dimensional");

  _batch_sizes = x.batch_sizes().vec();
  _options = x.options();

  _ndof = x.base_sizes()[0];
  _solution = x.clone();

  reinit_implicit_system(true, true, true);
}

void
TestNonlinearSystem::reinit_implicit_system(bool /*s*/, bool r, bool J)
{
  if (r)
    _residual = BatchTensor::zeros(_batch_sizes, {_ndof}, _options);

  if (J)
    _Jacobian = BatchTensor::zeros(_batch_sizes, {_ndof, _ndof}, _options);
}

PowerTestSystem::PowerTestSystem(const OptionSet & options)
  : TestNonlinearSystem(options)
{
}

void
PowerTestSystem::assemble(bool residual, bool Jacobian)
{
  if (residual)
    for (TorchSize i = 0; i < _ndof; i++)
      _residual.base_index_put({i},
                               math::pow(_solution.base_index({i}), Scalar(i + 1, _options)) - 1.0);

  if (Jacobian)
    for (TorchSize i = 0; i < _ndof; i++)
      _Jacobian.base_index_put({i, i},
                               (i + 1) * math::pow(_solution.base_index({i}), Scalar(i, _options)));
}

BatchTensor
PowerTestSystem::exact_solution() const
{
  return BatchTensor::ones(_batch_sizes, {_ndof}, _options);
}
