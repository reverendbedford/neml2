#pragma once

#include "NonlinearSystem.h"
#include "types.h"

/// Test system of equations, just also gives you the exact solution
class TestNonlinearSystem : public NonlinearSystem
{
public:
  virtual torch::Tensor exact_solution() const = 0;
  virtual torch::Tensor guess() const = 0;
};

/// Batched x**n for arbitrary n
class PowerTestSystem : public TestNonlinearSystem
{
public:
  PowerTestSystem(TorchSize nbatch, TorchSize n);

  virtual torch::Tensor residual(torch::Tensor x);
  virtual torch::Tensor jacobian(torch::Tensor x);
  virtual torch::Tensor exact_solution() const;
  virtual torch::Tensor guess() const;

private:
  TorchSize _nbatch;
  TorchSize _n;
};
