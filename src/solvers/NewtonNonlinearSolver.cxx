#include "solvers/NewtonNonlinearSolver.h"

NewtonNonlinearSolver::NewtonNonlinearSolver(const NonlinearSolverParameters & params)
  : NonlinearSolver(params)
{
}

torch::Tensor
NewtonNonlinearSolver::solve(const torch::Tensor & x0, NonlinearSystem & system) const
{
  // Setup initial guess and initial residual
  torch::Tensor x = x0;
  auto R = system.residual(x);
  auto nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);
  torch::Tensor nR0 = nR;

  // Begin iterating
  size_t i = 0;

  nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  while (torch::any(nR > atol()).item<bool>() && torch::any(nR / nR0 > rtol()).item<bool>() &&
         (i < miters()))
  {
    // Get the new value of x by Newton's method
    auto J = system.jacobian(x);
    #if (TORCH_VERSION_MINOR > 12)
    x -= torch::linalg::solve(J, R, true);
    #else
    x -= torch::linalg::solve(J, R);
    #endif

    // Update R and the norm of R
    R = system.residual(x);
    nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);

    // Iterate
    i += 1;
  }

  // Throw if we exceeded miters
  if (i == miters())
    throw std::runtime_error("Nonlinear solver exceeded miters!");

  return x;
}
