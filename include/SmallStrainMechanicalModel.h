#pragma once

#include "ConstitutiveModel.h"

/// Common ancestor for small kinematics mechanical material models
//
//  We should at least add temperature at some point...
//
class SmallStrainMechanicalModel : public ConstitutiveModel
{
public:
  /// Add stress to the state
  virtual StateInfo state() const;

  /// Any internal state
  virtual StateInfo internal_state() const = 0;

  /// Fully define the driving forces
  virtual StateInfo forces() const;
};
