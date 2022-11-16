#include "SmallStrainMechanicalModel.h"
#include "SymR2.h"

StateInfo
SmallStrainMechanicalModel::state() const
{
  StateInfo state;
  state.add<SymR2>("stress");
  return state;
}

void
SmallStrainMechanicalModel::initial_state(State & state) const
{
  state.set<SymR2>("stress", SymR2::zeros().expand_batch(state.tensor().batch_sizes()));
}

StateInfo
SmallStrainMechanicalModel::forces() const
{
  StateInfo forces;
  forces.add<SymR2>("strain");
  forces.add<Scalar>("time");
  return forces;
}
