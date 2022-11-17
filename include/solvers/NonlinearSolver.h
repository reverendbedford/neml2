#pragma once

#include <torch/torch.h>

#include "solvers/NonlinearSystem.h"
#include "types.h"

/// Parameter set provided to nonlinear solvers
struct NonlinearSolverParameters
{
  Real atol = 1e-8;
  Real rtol = 1e-6;
  size_t miters = 100;
};

/// Base class for a nonlinear solver
class NonlinearSolver
{
public:
  NonlinearSolver(const NonlinearSolverParameters & params);
  virtual torch::Tensor solve(const torch::Tensor & x0, NonlinearSystem & system) const = 0;
  virtual Real rtol() const { return _params.rtol; };
  virtual Real atol() const { return _params.atol; };
  virtual size_t miters() const { return _params.miters; };

protected:
  NonlinearSolverParameters _params;
};
