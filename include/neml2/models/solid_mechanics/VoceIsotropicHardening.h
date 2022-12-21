#pragma once

#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
class VoceIsotropicHardening : public IsotropicHardening
{
public:
  VoceIsotropicHardening(const std::string & name, Scalar R, Scalar d);

protected:
  /// Voce saturating hardening map
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _R, _d;
};
} // namespace neml2
