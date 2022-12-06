#pragma once

#include "models/solid_mechanics/KinematicHardening.h"

class NoKinematicHardening : public KinematicHardening
{
public:
  using KinematicHardening::KinematicHardening;

  /// No kinematic hardening
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
