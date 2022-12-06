#pragma once

#include "tensors/BatchTensor.h"

/// Abstract base class for a nonlinear system of equations
class NonlinearSystem
{
public:
  virtual void set_residual(BatchTensor<1> x,
                            BatchTensor<1> residual,
                            BatchTensor<1> * Jacobian = nullptr) const = 0;

  /// Convenient shortcut to construct and return the system residual
  virtual BatchTensor<1> residual(BatchTensor<1> in) const final;

  /// Convenient shortcut to construct and return the system Jacobian
  virtual BatchTensor<1> Jacobian(BatchTensor<1> in) const final;

  /// Convenient shortcut to construct and return the system residual and Jacobian
  virtual std::tuple<BatchTensor<1>, BatchTensor<1>>
  residual_and_Jacobian(BatchTensor<1> in) const final;
};
