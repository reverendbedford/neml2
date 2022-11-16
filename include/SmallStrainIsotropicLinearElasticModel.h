#pragma once

#include "SmallStrainMechanicalModel.h"

#include "SymSymR4.h"
#include "Scalar.h"

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
