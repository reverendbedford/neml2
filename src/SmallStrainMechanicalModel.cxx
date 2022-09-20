#include "SmallStrainMechanicalModel.h"

StateInfo SmallStrainMechanicalModel::state() const
{
  StateInfo state;
  state.add<BatchedSymR2>("stress");
  state.add_substate("internal_state", internal_state());
  return state;
}

StateInfo SmallStrainMechanicalModel::forces() const
{
  StateInfo forces;
  forces.add<BatchedSymR2>("strain");
  return forces;
}
