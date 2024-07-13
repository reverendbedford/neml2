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

namespace neml2
{
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
  _residual = BatchTensor::zeros(_batch_sizes, {_ndof}, _options);
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
    for (Size i = 0; i < _ndof; i++)
      _residual.base_index_put({i},
                               math::pow(_solution.base_index({i}), Scalar(i + 1, _options)) - 1.0);

  if (Jacobian)
    for (Size i = 0; i < _ndof; i++)
      _Jacobian.base_index_put({i, i},
                               (i + 1) * math::pow(_solution.base_index({i}), Scalar(i, _options)));
}

BatchTensor
PowerTestSystem::exact_solution() const
{
  return BatchTensor::ones(_batch_sizes, {_ndof}, _options);
}

RosenbrockTestSystem::RosenbrockTestSystem(const OptionSet & options)
  : TestNonlinearSystem(options)
{
}

void
RosenbrockTestSystem::assemble(bool residual, bool Jacobian)
{
  if (residual)
  {
    auto xm = _solution.base_index({indexing::Slice(1, -1)});
    auto xm_m1 = _solution.base_index({indexing::Slice(0, -2)});
    auto xm_p1 = _solution.base_index({indexing::Slice(2, indexing::None)});

    auto x0 = _solution.base_index({0});
    auto x1 = _solution.base_index({1});

    auto xn1 = _solution.base_index({-1});
    auto xn2 = _solution.base_index({-2});

    _residual.base_index_put({indexing::Slice(1, -1)},
                             200 * (xm - math::pow(xm_m1, 2.0)) -
                                 400 * (xm_p1 - math::pow(xm, 2.0)) * xm - 2 * (1 - xm));
    _residual.base_index_put({0}, -400 * x0 * (x1 - math::pow(x0, 2.0)) - 2 * (1 - x0));
    _residual.base_index_put({-1}, 200.0 * (xn1 - math::pow(xn2, 2.0)));
  }

  if (Jacobian)
  {
    auto s_x0n1 = _solution.base_index({indexing::Slice(0, -1)});
    auto s_x11 = _solution.base_index({indexing::Slice(1, -1)});
    auto s_x2 = _solution.base_index({indexing::Slice(2, indexing::None)});

    auto x0 = _solution.base_index({0});
    auto x1 = _solution.base_index({1});

    auto d1 = -400 * s_x0n1;
    auto H = torch::diag_embed(d1, -1) + torch::diag_embed(d1, 1);
    auto diagonal = BatchTensor::zeros_like(_solution);

    diagonal.base_index_put({0}, 1200 * math::pow(x0, 2.0) - 400.0 * x1 + 2);
    diagonal.base_index_put({-1}, Scalar(200.0, _options));
    diagonal.base_index_put({indexing::Slice(1, -1)},
                            202 + 1200 * math::pow(s_x11, 2.0) - 400 * s_x2);

    _Jacobian = BatchTensor(torch::diag_embed(diagonal) + H, _solution.batch_dim());
  }
}

BatchTensor
RosenbrockTestSystem::exact_solution() const
{
  return BatchTensor::ones(_batch_sizes, {_ndof}, _options);
}
}
