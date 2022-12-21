#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
IsotropicHardening::IsotropicHardening(const std::string & name)
  : Model(name),
    equivalent_plastic_strain(
        declareInputVariable<Scalar>({"state", "internal_state", "equivalent_plastic_strain"})),
    isotropic_hardening(
        declareOutputVariable<Scalar>({"state", "hardening_interface", "isotropic_hardening"}))
{
  setup();
}
} // namespace neml2
