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

#include "neml2/solvers/TrustRegionSubProblem.h"
#include "neml2/misc/math.h"

namespace neml2
{
TrustRegionSubProblem::TrustRegionSubProblem(const OptionSet & options)
  : NonlinearSystem(options)
{
}

void
TrustRegionSubProblem::reinit(const NonlinearSystem & system, const Scalar & delta)
{
  _batch_sizes = delta.batch_sizes().vec();
  _options = delta.options();

  _solution = Scalar::zeros(_batch_sizes, _options);
  _residual = Scalar::empty(_batch_sizes, _options);
  _Jacobian = Scalar::empty(_batch_sizes, _options);

  _R = system.residual_view().clone();
  _J = system.Jacobian_view().clone();
  _delta = delta.clone();

  _JJ = math::bmm(_J.base_transpose(0, 1), _J);
  _JR = math::bmv(_J.base_transpose(0, 1), _R);
}

void
TrustRegionSubProblem::assemble(bool residual, bool Jacobian)
{
  auto s = Scalar(_solution);
  auto p = -preconditioned_direction(s);
  auto np = math::sqrt(math::bvv(p, p));

  if (residual)
    _residual = 1.0 / np - 1.0 / math::sqrt(2.0 * _delta);

  if (Jacobian)
    _Jacobian = 1.0 / math::pow(np, 3.0) * math::bvv(p, preconditioned_solve(s, p));
}

BatchTensor
TrustRegionSubProblem::preconditioned_solve(const Scalar & s, const BatchTensor & v) const
{
  return math::linalg::solve(_JJ + s * BatchTensor::identity(v.base_sizes()[0], _options), v);
}

BatchTensor
TrustRegionSubProblem::preconditioned_direction(const Scalar & s) const
{
  return preconditioned_solve(s, _JR);
}
} // namespace neml2
