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

#include "neml2/solvers/Newton.h"
#include <iomanip>
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(Newton);

OptionSet
Newton::expected_options()
{
  OptionSet options = NonlinearSolver::expected_options();
  return options;
}

Newton::Newton(const OptionSet & options)
  : NonlinearSolver(options)
{
}

std::tuple<bool, size_t>
Newton::solve(NonlinearSystem & system, BatchTensor & x)
{
  neml_assert_dbg(!x.requires_grad(), "The trial solution shall not contain any function graph.");

  // The initial residual for relative convergence check
  system.residual();
  auto nR = system.residual_norm();
  auto nR0 = nR.clone();

  // Check for initial convergence
  if (converged(0, nR0, nR0))
  {
    // TODO: The final update is only necessary if we use AD
    system.Jacobian();
    final_update(system, x);
    return {true, 0};
  }

  // Prepare any solver internal data before the iterative update
  prepare(system, x);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < miters; i++)
  {
    system.Jacobian();
    update(system, x);
    system.residual();
    nR = system.residual_norm();

    // Check for convergence
    if (converged(i, nR, nR0))
    {
      // TODO: The final update is only necessary if we use AD
      system.Jacobian();
      final_update(system, x);
      return {true, i};
    }
  }

  return {false, miters};
}

bool
Newton::converged(size_t itr, const torch::Tensor & nR, const torch::Tensor & nR0) const
{
  // LCOV_EXCL_START
  if (verbose)
    std::cout << "ITERATION " << std::setw(3) << itr << ", |R| = " << std::scientific
              << torch::max(nR).item<Real>() << ", |R0| = " << std::scientific
              << torch::max(nR0).item<Real>() << std::endl;
  // LCOV_EXCL_STOP

  return torch::all(torch::logical_or(nR < atol, nR / nR0 < rtol)).item<bool>();
}

void
Newton::update(NonlinearSystem & system, BatchTensor & x)
{
  auto dx = solve_direction(system);

  x.variable_data() += system.scale_direction(dx);
  system.set_solution(x);
}

void
Newton::final_update(NonlinearSystem & system, BatchTensor & x)
{
  auto dx = solve_direction(system);
  x += system.scale_direction(dx);
}

BatchTensor
Newton::solve_direction(const NonlinearSystem & system)
{
  // Special case when this is a scalar system
  if (system.residual_view().base_dim() == 0)
    return -system.residual_view() / system.Jacobian_view();

  return -math::linalg::solve(system.Jacobian_view(), system.residual_view());
}

} // namespace neml2
