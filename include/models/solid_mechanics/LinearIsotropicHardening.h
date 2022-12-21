#pragma once

#include "models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
class LinearIsotropicHardening : public IsotropicHardening
{
public:
  LinearIsotropicHardening(const std::string & name, Scalar K);

protected:
  /// Simple linear map between equivalent strain and hardening
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _K;
};
} // namespace neml2
