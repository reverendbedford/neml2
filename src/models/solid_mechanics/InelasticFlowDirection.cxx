#include "models/solid_mechanics/InelasticFlowDirection.h"

StateInfo
InelasticFlowDirection::output() const
{
  StateInfo interface;
  interface.add<SymR2>("flow_direction");
  return interface;
}
