#pragma once

#include <torch/torch.h>

#include "neml2/base/ParameterSet.h"

namespace neml2
{
class NEML2Object : public torch::nn::Module
{
public:
  static ParameterSet expected_params();

  NEML2Object(const std::string & name);

  NEML2Object(const ParameterSet & params);
};
} // namespace neml2
