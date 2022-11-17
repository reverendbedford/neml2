#include "models/solid_mechanics/IsotropicHardeningMap.h"

StateInfo
IsotropicHardeningMap::state() const
{
  StateInfo state;
  state.add<Scalar>("equivalent_plastic_strain");
  return state;
}

void
IsotropicHardeningMap::initial_state(State & input) const
{
  input.set<Scalar>("equivalent_plastic_strain", Scalar(0, input.batch_size()));
}

StateInfo
IsotropicHardeningMap::output() const
{
  StateInfo res;
  StateInfo stress;
  stress.add<SymR2>("stress");
  res.add_substate("stress_interface", stress);
  StateInfo hardening;
  hardening.add<Scalar>("isotropic_hardening");
  res.add_substate("hardening_interface", hardening);
  return res;
}

std::string
IsotropicHardeningMap::conjugate_name(std::string stress_var) const
{
  if (stress_var == "isotropic_hardening")
    return "equivalent_plastic_strain";
  else
    return stress_var;
}
