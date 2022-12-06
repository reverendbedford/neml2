#pragma once

#include "solvers/NonlinearSystem.h"

/// Test system of equations, just also gives you the exact solution
class TestNonlinearSystem : public NonlinearSystem
{
public:
  virtual BatchTensor<1> exact_solution(BatchTensor<1> x) const = 0;
  virtual BatchTensor<1> guess(BatchTensor<1> x) const = 0;
};

/// Batched x**n for arbitrary n
class PowerTestSystem : public TestNonlinearSystem
{
public:
  PowerTestSystem();

  virtual void set_residual(BatchTensor<1> x,
                            BatchTensor<1> residual,
                            BatchTensor<1> * Jacobian = nullptr) const;
  virtual BatchTensor<1> exact_solution(BatchTensor<1> x) const;
  virtual BatchTensor<1> guess(BatchTensor<1> x) const;
};
