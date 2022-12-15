#include "models/solid_mechanics/KinematicHardening.h"

namespace neml2
{
KinematicHardening::KinematicHardening(const std::string & name)
  : Model(name),
    cauchy_stress(declareInputVariable<SymR2>("state", "cauchy_stress")),
    mandel_stress(declareOutputVariable<SymR2>("state", "mandel_stress"))
{
  setup();
}
} // namespace neml2
