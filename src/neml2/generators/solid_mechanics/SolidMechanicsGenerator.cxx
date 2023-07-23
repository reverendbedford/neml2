// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/generators/solid_mechanics/SolidMechanicsGenerator.h"

namespace neml2
{
register_NEML2_generator("SolidMechanics", SolidMechanicsGenerator);

ParameterSet
SolidMechanicsGenerator::expected_params()
{
  ParameterSet params = Generator::expected_params();
  return params;
}

SolidMechanicsGenerator::SolidMechanicsGenerator(hit::Node * root)
  : Generator(root),
    _known_mechanisms({"Elasticity", "Viscoplasticity"})
{
}

ParameterCollection
SolidMechanicsGenerator::generate()
{
  ParameterCollection params;

  // Find all the mechanisms
  for (auto node : _root->children(hit::NodeType::Section))
  {
    auto name = node->path();
    neml_assert(_known_mechanisms.count(name), "Unknown mechanism in solid mechanics: ", name);
    _mechanisms[name] = node;
  }

  params.merge(generate_elasticity());

  return params;
}

ParameterCollection
SolidMechanicsGenerator::generate_elasticity()
{
  neml_assert(_mechanisms.count("Elasticity"), "Solid mechanics must define an elasticity model.");

  ParameterCollection params;
  hit::Node * node = _mechanisms["Elasticity"];

  // Hidden object name
  std::string name = "_elasticity";

  // There is a special field reserved for object type
  std::string type = node->param<std::string>("type");

  // Retrieve the expected parameters of this object
  ParameterSet emodel_params = Registry::expected_params(type);
  emodel_params.set<std::string>("name") = name;
  emodel_params.set<std::string>("type") = type;

  // Extract other parameters
  ExtractParamsWalker epw(emodel_params);
  node->walk(&epw);
  params["Models"][name] = emodel_params;

  return params;
}
} // namespace neml2
