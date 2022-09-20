#pragma once

#include "SmallStrainMechanicalModel.h"

#include "Scalar.h"

/// Simplest test model...
class SmallStrainIsotropicLinearElasticModel: public SmallStrainMechanicalModel {
 public:
  SmallStrainIsotropicLinearElasticModel(const Scalar & E, 
                                         const Scalar & nu);

  /// Update the state to next time step
  virtual void update(State & state_np1, const State & forces_np1,
                      const State & state_n, const State & forces_n);

  /// Trivial implementation of internal_state (nothing)
  virtual StateInfo internal_state() const;

 protected:
  Scalar _E, _nu;
};
