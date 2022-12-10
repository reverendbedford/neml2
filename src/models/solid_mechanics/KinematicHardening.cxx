#include "models/solid_mechanics/KinematicHardening.h"

namespace neml2
{
KinematicHardening::KinematicHardening(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<SymR2>("cauchy_stress");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<SymR2>("mandel_stress");

  setup();
}
} // namespace neml2
