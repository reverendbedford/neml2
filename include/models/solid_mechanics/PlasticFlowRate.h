#pragma once

#include "models/Model.h"

namespace neml2
{
/// Defines the consistency parameter
class PlasticFlowRate : public Model
{
public:
  PlasticFlowRate(const std::string & name);
};
} // namespace neml2
