#pragma once

#include "models/solid_mechanics/YieldSurface.h"

class J2IsotropicYieldSurface : public YieldSurface
{
public:
  J2IsotropicYieldSurface() = default;

  /// The hardening quantities the surface expects
  virtual StateInfo hardening_interface() const;

  /// The value of the yield surface
  virtual Scalar f(const State & interface) const;
  /// The derivative of the surface with respect to the interface values
  virtual State df_ds(const State & interface) const;
  /// The second derivative of the surface with respect to the values
  virtual StateDerivative d2f_ds2(const State & interface) const;
};
