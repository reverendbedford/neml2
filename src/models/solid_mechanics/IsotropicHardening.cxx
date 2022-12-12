#include "models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
IsotropicHardening::IsotropicHardening(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<Scalar>("equivalent_plastic_strain");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("isotropic_hardening");

  setup();
}

IsotropicHardening::IsotropicHardening(InputParameters & params)
  : Model(params)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<Scalar>("equivalent_plastic_strain");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("isotropic_hardening");

  setup();
}
} // namespace neml2
