#pragma once

#include "models/Model.h"

namespace neml2
{
/// Defines the direction of plastic flow
class PlasticFlowDirection : public Model
{
public:
  PlasticFlowDirection(const std::string & name);
};
} // namespace neml2
