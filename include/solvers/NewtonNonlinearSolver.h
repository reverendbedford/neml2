#pragma once

#include "solvers/NonlinearSolver.h"

/// Direct torch implementation of Newton-Raphson
class NewtonNonlinearSolver : public NonlinearSolver
{
public:
  NewtonNonlinearSolver(const NonlinearSolverParameters & params);

  virtual BatchTensor<1> solve(const NonlinearSystem & system, const BatchTensor<1> & x0) const;

protected:
  virtual void update(BatchTensor<1> x, BatchTensor<1> R, BatchTensor<1> J) const;

  virtual bool converged(size_t itr, BatchTensor<1> nR, BatchTensor<1> nR0) const;
};
