#include "models/solid_mechanics/PlasticFlowRate.h"

namespace neml2
{
PlasticFlowRate::PlasticFlowRate(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<Scalar>("yield_function");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("hardening_rate");
  setup();
}
} // namespace neml2
