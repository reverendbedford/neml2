#include "neml2/base/NEML2Object.h"

namespace neml2
{
ParameterSet
NEML2Object::expected_params()
{
  ParameterSet params;
  params.set<std::string>("name");
  return params;
}

NEML2Object::NEML2Object(const std::string & name)
  : torch::nn::Module(name)
{
}

NEML2Object::NEML2Object(const ParameterSet & params)
  : torch::nn::Module(params.get<std::string>("name"))
{
}
} // namespace neml2
