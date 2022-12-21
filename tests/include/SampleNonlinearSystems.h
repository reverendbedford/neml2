#pragma once

#include "neml2/solvers/NonlinearSystem.h"

/// Test system of equations, just also gives you the exact solution
class TestNonlinearSystem : public neml2::NonlinearSystem
{
public:
  virtual neml2::BatchTensor<1> exact_solution(neml2::BatchTensor<1> x) const = 0;
  virtual neml2::BatchTensor<1> guess(neml2::BatchTensor<1> x) const = 0;
};

/// Batched x**n for arbitrary n
class PowerTestSystem : public TestNonlinearSystem
{
public:
  PowerTestSystem();

  virtual void set_residual(neml2::BatchTensor<1> x,
                            neml2::BatchTensor<1> residual,
                            neml2::BatchTensor<1> * Jacobian = nullptr) const;
  virtual neml2::BatchTensor<1> exact_solution(neml2::BatchTensor<1> x) const;
  virtual neml2::BatchTensor<1> guess(neml2::BatchTensor<1> x) const;
};
