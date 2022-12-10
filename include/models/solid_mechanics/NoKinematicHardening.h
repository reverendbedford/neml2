#pragma once

#include "models/solid_mechanics/KinematicHardening.h"

namespace neml2
{
class NoKinematicHardening : public KinematicHardening
{
public:
  using KinematicHardening::KinematicHardening;

protected:
  /// No kinematic hardening
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
