#pragma once

#include "models/solid_mechanics/SmallStrainMechanicalModel.h"

#include "tensors/SymSymR4.h"
#include "tensors/Scalar.h"

/// Simplest test model...
class SmallStrainIsotropicLinearElasticModel : public SmallStrainMechanicalModel
{
public:
  SmallStrainIsotropicLinearElasticModel(const Scalar & E, const Scalar & nu);

  /// Stress update
  virtual State value(StateInput input);
  /// Derivative of stress update
  virtual StateDerivativeOutput dvalue(StateInput input);

protected:
  Scalar _E, _nu;
  SymSymR4 _C;
};
