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

namespace neml2
{
NewtonNonlinearSolver::NewtonNonlinearSolver(const NonlinearSolverParameters & params)
  : NonlinearSolver(params)
{
}

BatchTensor<1>
NewtonNonlinearSolver::solve(const NonlinearSystem & system, const BatchTensor<1> & x0) const
{
  // Setup initial guess and initial residual
  BatchTensor<1> x = x0.clone();
  BatchTensor<1> R = system.residual(x);
  BatchTensor<1> nR0 = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

  // Check for initial convergence
  if (converged(0, nR0, nR0))
    return x;

  // Begin iterating
  BatchTensor<1> J = system.Jacobian(x);
  update(x, R, J);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < params.miters; i++)
  {
    // Update R and the norm of R
    std::tie(R, J) = system.residual_and_Jacobian(x);
    BatchTensor<1> nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

    // Check for initial convergence
    if (converged(i, nR, nR0))
      return x;

    update(x, R, J);
  }

  // Throw if we exceeded miters
  throw std::runtime_error("Nonlinear solver exceeded miters!");

  return x;
}

void
NewtonNonlinearSolver::update(BatchTensor<1> x, BatchTensor<1> R, BatchTensor<1> J) const
{
#if (TORCH_VERSION_MINOR > 12)
  x -= torch::linalg::solve(J, R, true);
#else
  x -= torch::linalg::solve(J, R);
#endif
}

bool
NewtonNonlinearSolver::converged(size_t itr, BatchTensor<1> nR, BatchTensor<1> nR0) const
{
  if (params.verbose)
    std::cout << "ITERATION " << std::setw(3) << itr << ", |R| = " << std::scientific
              << torch::mean(nR).item<double>() << ", |R0| = " << std::scientific
              << torch::mean(nR0).item<double>() << std::endl;

  return torch::all(nR < params.atol).item<bool>() ||
         torch::all(nR / nR0 < params.rtol).item<bool>();
}
} // namespace neml2
