#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
/// Defines the direction of plastic flow
class PlasticFlowDirection : public Model
{
public:
  PlasticFlowDirection(const std::string & name);

  const LabeledAxisAccessor plastic_flow_direction;
};
} // namespace neml2
