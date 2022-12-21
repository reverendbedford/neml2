#include "neml2/models/solid_mechanics/PlasticHardening.h"

namespace neml2
{
PlasticHardening::PlasticHardening(const std::string & name)
  : Model(name),
    hardening_rate(declareInputVariable<Scalar>({"state", "hardening_rate"}))
{
  setup();
}
} // namespace neml2
