#include "models/solid_mechanics/KinematicHardening.h"

namespace neml2
{
KinematicHardening::KinematicHardening(const std::string & name)
  : Model(name),
    plastic_strain(declareInputVariable<SymR2>({"state", "internal_state", "plastic_strain"})),
    kinematic_hardening(
        declareOutputVariable<SymR2>({"state", "hardening_interface", "kinematic_hardening"}))
{
  setup();
}
} // namespace neml2
