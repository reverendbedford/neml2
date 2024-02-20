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

#pragma once

#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
/**
 * The trust region subproblem introduced in
 *
 * > Yuan, Ya-xiang. Trust region algorithms for nonlinear equations. Hong Kong Baptist
 * > University, Department of Mathematics, 1994.
 */
class TrustRegionSubProblem : public NonlinearSystem
{
public:
  TrustRegionSubProblem(const OptionSet & options);

  virtual void reinit(const NonlinearSystem & system, const Scalar & delta);

  BatchTensor preconditioned_direction(const Scalar & s) const;

protected:
  virtual void assemble(bool residual, bool Jacobian) override;

  BatchTensor preconditioned_solve(const Scalar & s, const BatchTensor & v) const;

  TorchShape _batch_sizes;

  torch::TensorOptions _options;

private:
  /// Residual of the underlying nonlinear problem
  BatchTensor _R;

  /// Jacobian of the underlying nonlinear problem
  BatchTensor _J;

  /// The trust region radius
  Scalar _delta;

  /// Temporary Jacobian-Jacobian product
  BatchTensor _JJ;

  /// Temporary Jacobian-Residual product
  BatchTensor _JR;
};
} // namespace neml2
