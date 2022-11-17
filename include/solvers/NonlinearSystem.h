#pragma once

#include <torch/torch.h>

/// Abstract base class for a nonlinear system of equations
class NonlinearSystem
{
public:
  virtual torch::Tensor residual(torch::Tensor x) = 0;
  virtual torch::Tensor jacobian(torch::Tensor x) = 0;
};
