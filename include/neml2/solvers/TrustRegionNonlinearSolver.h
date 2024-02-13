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

#include "neml2/solvers/NonlinearSolver.h"

namespace neml2
{
/**
 * @copydoc neml2::NonlinearSolver
 *
 * Trust region nonlinear solver
 */
class TrustRegionNonlinearSolver : public NonlinearSolver
{
public:
  static OptionSet expected_options();

  TrustRegionNonlinearSolver(const OptionSet & options);

  virtual BatchTensor solve(const NonlinearSystem & system, const BatchTensor & x0) const override;

protected:
  /// Check for convergence and optionally print out
  virtual bool
  converged(size_t itr, const torch::Tensor & nR, const torch::Tensor & nR0, Real min_trust) const;

  /// Approximately solve the trust region subproblem and return the trial step
  BatchTensor
  solve_subproblem(const BatchTensor & R, const BatchTensor & J, const Scalar & delta) const;

  /// Potentially move this out to a separate object at some point
  BatchTensor
  scalar_newton(std::function<std::tuple<BatchTensor, BatchTensor>(const BatchTensor &)> RJ,
                const BatchTensor & x0) const;

  /// A helper method to calculate the trust region problem
  static BatchTensor
  JJsJ_product(const BatchTensor & J, const Scalar & sigma, const BatchTensor & v);

  /// A helper method to calculate the subproblem residual and jacobian
  static std::tuple<BatchTensor, BatchTensor>
  subproblem(const Scalar & s, const BatchTensor & R, const BatchTensor & J, const Scalar & delta);

  /// A helper to calculate the reduction in the merit function
  static Scalar
  merit_function_reduction(const BatchTensor & p, const BatchTensor & R, const BatchTensor & J);

  /// Initial size of the trust region
  Real _delta_0;

  /// Maximum size of the trust region
  Real _delta_max;

  /// Criteria for reducing the trust region
  Real _reduce_criteria;

  /// Criteria for expanding the trust region
  Real _expand_criteria;

  /// Cutback factor if we do reduce the trust region
  Real _reduce_factor;

  /// Expansion factor if we do increase the trust region
  Real _expand_factor;

  /// Acceptance criteria for a step
  Real _accept_criteria;

  /// Relative tolerance for scalar subproblem solve
  Real _subproblem_rtol;

  /// Absolute tolerance for the scalar subproblem solve
  Real _subproblem_atol;

  /// Maximum iterations for the scalar subproblem solve
  unsigned int _subproblem_miter;
};
} // namespace neml2
