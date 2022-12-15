#include "models/solid_mechanics/PlasticFlowDirection.h"

namespace neml2
{
PlasticFlowDirection::PlasticFlowDirection(const std::string & name)
  : Model(name),
    plastic_flow_direction(declareOutputVariable<SymR2>("state", "plastic_flow_direction"))
{
  setup();
}
} // namespace neml2
