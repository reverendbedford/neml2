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

#include "neml2/solvers/NewtonNonlinearSolver.h"
#include <iomanip>
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(NewtonNonlinearSolver);

OptionSet
NewtonNonlinearSolver::expected_options()
{
  OptionSet options = NonlinearSolver::expected_options();
  options.set<bool>("linesearch") = false;
  options.set<unsigned int>("max_linesearch_iterations") = 10;
  options.set<Real>("linesearch_cutback") = 2.0;
  options.set<Real>("linesearch_stopping_criteria") = 1.0e-3;
  return options;
}

NewtonNonlinearSolver::NewtonNonlinearSolver(const OptionSet & options)
  : NonlinearSolver(options),
    _linesearch(options.get<bool>("linesearch")),
    _linesearch_miter(options.get<unsigned int>("max_linesearch_iterations")),
    _linesearch_sigma(options.get<Real>("linesearch_cutback")),
    _linesearch_c(options.get<Real>("linesearch_stopping_criteria"))
{
}

std::tuple<bool, size_t>
NewtonNonlinearSolver::solve(NonlinearSystem & system) const
{
  // The trial solution that we are going to update
  auto x = system.solution().clone();

  // The initial residual for relative convergence check
  auto R = system.residual();
  auto nR0 = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

  // Check for initial convergence
  if (converged(0, nR0, nR0, 1.0))
    return {true, 0};

  // The line search parameter (1 for full Newton step)
  Real alpha;

  // Begin iterating
  auto J = system.Jacobian();
  alpha = update(system, x, R, J);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < miters; i++)
  {
    std::tie(R, J) = system.residual_and_Jacobian(x);
    auto nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

    // Check for initial convergence
    if (converged(i, nR, nR0, alpha))
      return {true, i};

    // Update trial solution
    alpha = update(system, x, R, J);
  }

  return {false, miters};
}

bool
NewtonNonlinearSolver::converged(size_t itr,
                                 const torch::Tensor & nR,
                                 const torch::Tensor & nR0,
                                 Real alpha) const
{
  // LCOV_EXCL_START
  if (verbose)
    std::cout << "ITERATION " << std::setw(3) << itr << ", |R| = " << std::scientific
              << torch::max(nR).item<Real>() << ", |R0| = " << std::scientific
              << torch::max(nR0).item<Real>() << ", alpha = " << std::scientific << alpha
              << std::endl;
  // LCOV_EXCL_STOP

  return torch::all(nR < atol).item<bool>() || torch::all(nR / nR0 < rtol).item<bool>();
}

Real
NewtonNonlinearSolver::update(NonlinearSystem & system,
                              BatchTensor & x,
                              const BatchTensor & R,
                              const BatchTensor & J) const
{
  auto dx = solve_direction(system, R, J);
  Real alpha = _linesearch ? linesearch(system, x, R.clone(), dx) : 1.0;
  x += alpha * dx;
  return alpha;
}

Real
NewtonNonlinearSolver::linesearch(NonlinearSystem & system,
                                  const BatchTensor & x,
                                  const BatchTensor & R0,
                                  const BatchTensor & dx) const
{
  Real alpha = 1.0;
  auto nR02 = torch::linalg_vecdot(R0, R0);

  for (size_t i = 1; i < _linesearch_miter; i++)
  {
    auto R = system.residual(x + alpha * dx);
    auto nR2 = torch::linalg_vecdot(R, R);
    auto crit = nR02 + 2.0 * _linesearch_c * alpha * torch::linalg_vecdot(R0, dx);
    if (verbose)
      std::cout << "     LS ITERATION " << std::setw(3) << i << ", |R| = " << std::scientific
                << torch::max(torch::sqrt(nR2)).item<Real>() << ", |Rc| = " << std::scientific
                << torch::min(torch::sqrt(crit)).item<Real>() << std::endl;

    if (torch::all(nR2 <= crit).item<bool>() || torch::all(nR2 <= std::pow(atol, 2)).item<bool>())
      break;
    alpha /= _linesearch_sigma;
  }

  return alpha;
}

BatchTensor
NewtonNonlinearSolver::solve_direction(NonlinearSystem & system,
                                       const BatchTensor & R,
                                       const BatchTensor & J) const
{
  auto p = -BatchTensor(torch::linalg::solve(J, R, true), R.batch_dim());
  return system.scale_direction(p);
}

} // namespace neml2
