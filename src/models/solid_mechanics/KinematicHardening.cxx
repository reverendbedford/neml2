#include "models/solid_mechanics/KinematicHardening.h"

namespace neml2
{
KinematicHardening::KinematicHardening(const std::string & name)
  : Model(name),
    _cauchy_idx(declareVariable<SymR2>(input(), "state", "cauchy_stress")),
    _mandel_idx(declareVariable<SymR2>(output(), "state", "mandel_stress"))
{
  setup();
}
} // namespace neml2
