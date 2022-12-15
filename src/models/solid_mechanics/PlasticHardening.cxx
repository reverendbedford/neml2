#include "models/solid_mechanics/PlasticHardening.h"

namespace neml2
{
PlasticHardening::PlasticHardening(const std::string & name)
  : Model(name),
    hardening_rate(declareInputVariable<Scalar>({"state", "hardening_rate"})),
    equivalent_plastic_strain_rate(
        declareOutputVariable<Scalar>({"state", "equivalent_plastic_strain_rate"}))
{
  setup();
}
} // namespace neml2
