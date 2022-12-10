#include "models/solid_mechanics/PlasticFlowDirection.h"

namespace neml2
{
PlasticFlowDirection::PlasticFlowDirection(const std::string & name)
  : Model(name)
{
  output().add<LabeledAxis>("state");
  output().subaxis("state").add<SymR2>("plastic_flow_direction");
  setup();
}
} // namespace neml2
