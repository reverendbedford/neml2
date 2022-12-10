#pragma once

#include "solvers/NonlinearSystem.h"

namespace neml2
{
/// Parameter set provided to nonlinear solvers
struct NonlinearSolverParameters
{
  Real atol = 1e-8;
  Real rtol = 1e-6;
  size_t miters = 100;
  bool verbose = false;
};

/// Base class for a nonlinear solver
class NonlinearSolver
{
public:
  NonlinearSolver(const NonlinearSolverParameters & p);

  virtual BatchTensor<1> solve(const NonlinearSystem & system, const BatchTensor<1> & x0) const = 0;

  NonlinearSolverParameters params;
};
} // namespace neml2
