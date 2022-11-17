#pragma once

#include "models/solid_mechanics/InelasticFlowDirection.h"
#include "models/solid_mechanics/InelasticFlowRate.h"
#include "models/solid_mechanics/InelasticHardening.h"
#include "models/RateForm.h"
#include "models/solid_mechanics/SmallStrainMechanicalModel.h"
#include "tensors/SymSymR4.h"

/// Standard structural inelastic model
//  \dot{\sigma} = C : (\dot{\varepsilon} - \dot{\varepsilon}_p)
//  \dot{\varepsilon}_p = InelasticFlowRate * InelasticFlowDirection
//  \dot{h} = InelasticHardening
class InelasticModel : public RateForm, public SmallStrainMechanicalModel
{
public:
  InelasticModel(SymSymR4 C,
                 InelasticFlowRate & rate,
                 InelasticFlowDirection & direction,
                 InelasticHardening & hardening);

  /// Add internal variables to state
  virtual StateInfo state() const;
  /// Initialize internal variables
  virtual void initial_state(State & state) const;

  /// Outputs
  virtual StateInfo output() const;

  /// Forces
  virtual StateInfo forces() const;

  /// State rate
  virtual State value(StateInput input);
  /// Derivative of the state rate with respect to state and forces_rate
  virtual StateDerivativeOutput dvalue(StateInput input);

protected:
  SymSymR4 _C;
  InelasticFlowRate & _rate;
  InelasticFlowDirection & _direction;
  InelasticHardening & _hardening;
};
