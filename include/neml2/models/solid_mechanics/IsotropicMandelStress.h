#pragma once

#include "neml2/models/solid_mechanics/MandelStress.h"

namespace neml2
{
class IsotropicMandelStress : public MandelStress
{
public:
  using MandelStress::MandelStress;

protected:
  /// No kinematic hardening
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
