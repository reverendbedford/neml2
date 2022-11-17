#pragma once

#include "models/solid_mechanics/IsotropicHardeningMap.h"

class LinearIsotropicHardening : public IsotropicHardeningMap
{
public:
  LinearIsotropicHardening(Scalar s0, Scalar K);

  /// Simple linear map between equivalent strain and hardening
  virtual State value(State input);
  /// Derivative of the hardening map
  virtual StateDerivative dvalue(State input);

protected:
  Scalar _s0, _K;
};
