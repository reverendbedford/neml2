#pragma once

#include "tensors/Scalar.h"
#include "tensors/SymR2.h"
#include "tensors/SymSymR4.h"
#include "state/State.h"
#include "state/StateDerivative.h"

/// Parent class for all yield surfaces
class YieldSurface : public torch::nn::Module
{
public:
  /// The interface to the yield surface: what quantities it expects
  virtual StateInfo interface() const;

  /// The hardening quantities the surface expects
  virtual StateInfo hardening_interface() const = 0;

  /// The value of the yield surface
  virtual Scalar f(const State & interface) const = 0;
  /// The derivative of the surface with respect to the interface values
  virtual State df_ds(const State & interface) const = 0;
  /// The second derivative of the surface with respect to the values
  virtual StateDerivative d2f_ds2(const State & interface) const = 0;
};
