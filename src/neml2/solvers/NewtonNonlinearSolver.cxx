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

BatchTensor
NewtonNonlinearSolver::solve(const NonlinearSystem & system, const BatchTensor & x0) const
{
  // Setup initial guess and initial residual
  auto x = x0.clone();
  auto R = system.residual(x);
  auto nR0 = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

  // Check for initial convergence
  if (converged(0, nR0, nR0, 1.0))
    return x;

  Real alpha;
  // Begin iterating
  if (_linesearch)
    alpha = update_linesearch(x, R, system);
  else
    alpha = update_no_linesearch(x, R, system);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < miters; i++)
  {
    auto nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

    // Check for initial convergence
    if (converged(i, nR, nR0, alpha))
      return x;

    // Update R and the norm of R
    if (_linesearch)
      alpha = update_linesearch(x, R, system);
    else
      alpha = update_no_linesearch(x, R, system);
  }

  // Throw if we exceeded miters
  throw NEMLException("Nonlinear solver exceeded miters!");

  return x;
}

BatchTensor
NewtonNonlinearSolver::solve_direction(const NonlinearSystem & system,
                                       const BatchTensor & R,
                                       const BatchTensor & J) const
{
  auto p = BatchTensor(torch::linalg::solve(J, R, true), R.batch_dim());
  return system.scale_direction(p);
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

  return torch::all(torch::logical_or(nR < atol, nR / nR0 < rtol)).item<bool>();
}

Real
NewtonNonlinearSolver::update_no_linesearch(BatchTensor & x,
                                            BatchTensor & R,
                                            const NonlinearSystem & system) const
{
  BatchTensor J;
  std::tie(R, J) = system.residual_and_Jacobian(x);
  auto dx = solve_direction(system, R, J);
  x -= dx;
  return 1.0;
}

Real
NewtonNonlinearSolver::update_linesearch(BatchTensor & x,
                                         BatchTensor & R,
                                         const NonlinearSystem & system) const
{
  BatchTensor J;
  std::tie(R, J) = system.residual_and_Jacobian(x);

  auto R0 = R.clone();
  auto nR02 = torch::linalg_vecdot(R0, R0);
  auto alpha = Scalar::ones(x.batch_sizes(), x.options());

  auto dx = solve_direction(system, R, J);

  for (size_t i = 1; i < _linesearch_miter; i++)
  {
    R = system.residual(x - alpha * dx);
    auto nR2 = torch::linalg_vecdot(R, R);
    auto crit =
        nR02 + 2.0 * _linesearch_c * alpha * Scalar(torch::linalg_vecdot(R0, dx), x.batch_dim());
    if (verbose)
      std::cout << "     LS ITERATION " << std::setw(3) << i << ", |R| = " << std::scientific
                << torch::max(torch::sqrt(nR2)).item<Real>() << ", |Rc| = " << std::scientific
                << torch::min(torch::sqrt(crit)).item<Real>() << std::endl;

    auto stop = torch::logical_or(nR2 <= crit, nR2 <= std::pow(atol, 2));
    if (torch::all(stop).item<bool>())
      break;
    alpha.batch_index_put({torch::logical_not(stop)},
                          alpha.batch_index({torch::logical_not(stop)}) / _linesearch_sigma);
  }
  x -= alpha * dx;

  return torch::min(alpha).item<Real>();
}

} // namespace neml2
