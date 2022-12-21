#pragma once

#include "models/solid_mechanics/KinematicHardening.h"

namespace neml2
{
class LinearKinematicHardening : public KinematicHardening
{
public:
  LinearKinematicHardening(const std::string & name, Scalar H);

protected:
  /// Simple linear map between equivalent strain and hardening
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _H;
};
} // namespace neml2
