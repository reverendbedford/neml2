#include "models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
IsotropicHardening::IsotropicHardening(const std::string & name)
  : Model(name),
    _ep_idx(declareVariable<Scalar>(input(), "state", "equivalent_plastic_strain")),
    _g_idx(declareVariable<Scalar>(output(), "state", "isotropic_hardening"))
{
  setup();
}
} // namespace neml2
