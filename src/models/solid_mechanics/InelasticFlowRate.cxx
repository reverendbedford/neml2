#include "models/solid_mechanics/InelasticFlowRate.h"

StateInfo
InelasticFlowRate::output() const
{
  StateInfo output;
  output.add<Scalar>("flow_rate");
  return output;
}
