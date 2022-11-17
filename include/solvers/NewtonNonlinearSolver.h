#pragma once

#include "solvers/NonlinearSolver.h"
#include "tensors/Scalar.h"

/// Direct torch implementation of Newton-Raphson
class NewtonNonlinearSolver : public NonlinearSolver
{
public:
  NewtonNonlinearSolver(const NonlinearSolverParameters & params);
  virtual torch::Tensor solve(const torch::Tensor & x0, NonlinearSystem & system) const;
};
