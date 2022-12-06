#include "models/solid_mechanics/PlasticFlowRate.h"

PlasticFlowRate::PlasticFlowRate(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<Scalar>("yield_function");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("hardening_rate");
  setup();
}
