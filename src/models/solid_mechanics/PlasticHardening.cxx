#include "models/solid_mechanics/PlasticHardening.h"

namespace neml2
{
PlasticHardening::PlasticHardening(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<Scalar>("hardening_rate");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("equivalent_plastic_strain_rate");

  setup();
}
} // namespace neml2
