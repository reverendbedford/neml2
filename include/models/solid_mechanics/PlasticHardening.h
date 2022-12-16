#pragma once

#include "models/Model.h"

namespace neml2
{
/// Defines the hardening rate
class PlasticHardening : public Model
{
public:
  PlasticHardening(const std::string & name);

  const LabeledAxisAccessor hardening_rate;
};
} // namespace neml2
