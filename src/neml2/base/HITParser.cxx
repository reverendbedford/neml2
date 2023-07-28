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

#include "neml2/base/HITParser.h"
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/base/CrossRef.h"
#include "neml2/generators/Generator.h"
#include "neml2/base/GeneratorFactory.h"
#include <memory>

namespace neml2
{
ParameterCollection
HITParser::parse(const std::string & filename) const
{
  // Open and read the file
  std::ifstream file(filename);
  neml_assert(file.is_open(), "Unable to open file ", filename);

  // Read the file into a string
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string input = buffer.str();

  // Let HIT lex the string
  auto root = dynamic_cast<hit::Section *>(hit::parse("Hit parser", input));
  neml_assert(root, "HIT failed to lex the input file: ", filename);

  // First run the generators to generate parameters
  auto all_params = GeneratorFactory::get().generate(root);
  std::cout << all_params << std::endl;

  // Loop over each known section and extract parameters for each object
  for (const auto & section : Factory::pipeline)
  {
    auto section_node = root->find(section);
    if (section_node)
    {
      auto objects = section_node->children(hit::NodeType::Section);
      for (auto object : objects)
      {
        auto params = extract_object_parameters(object);
        all_params[section][params.get<std::string>("name")] = params;
      }
    }
  }

  return all_params;
}

ParameterSet
HITParser::extract_object_parameters(hit::Node * object) const
{
  // The object name is its node path
  std::string name = object->path();
  // There is a special field reserved for object type
  std::string type = object->param<std::string>("type");
  // Extract the parameters
  auto params = Registry::expected_params(type);
  extract_parameters(object, params);
  // Also fill in the special fields, e.g., name, type
  params.set<std::string>("name") = name;
  params.set<std::string>("type") = type;

  return params;
}

ParameterSet
HITParser::extract_generator_parameters(hit::Node * object) const
{
  auto params = GeneratorRegistry::expected_params(object->fullpath());
  for (auto node : object->children(hit::NodeType::Field))
    extract_parameter(node, params);
  return params;
}

void
HITParser::extract_parameters(hit::Node * object, ParameterSet & params) const
{
  for (auto node : object->children(hit::NodeType::Field))
    extract_parameter(node, params);
}

void
HITParser::extract_parameter(hit::Node * n, ParameterSet & params) const
{
#define extract_param_base(ptype, method)                                                          \
  else if (param->type() ==                                                                        \
           utils::demangle(                                                                        \
               typeid(ptype).name())) dynamic_cast<ParameterSet::Parameter<ptype> *>(param.get())  \
      ->set() = method(n->strVal())

#define extract_param_t(ptype)                                                                     \
  extract_param_base(ptype, utils::parse<ptype>);                                                  \
  extract_param_base(std::vector<ptype>, utils::parse_vector<ptype>);                              \
  extract_param_base(std::vector<std::vector<ptype>>, utils::parse_vector_vector<ptype>)

  if (n->type() == hit::NodeType::Field)
  {
    bool found = false;
    for (auto & [name, param] : params)
      if (name == n->path())
      {
        found = true;

        if (false)
          ;
        extract_param_t(bool);
        extract_param_t(int);
        extract_param_t(unsigned int);
        extract_param_t(TorchSize);
        extract_param_t(Real);
        extract_param_t(std::string);
        extract_param_t(LabeledAxisAccessor);
        extract_param_t(CrossRef<torch::Tensor>);
        extract_param_t(CrossRef<Scalar>);
        extract_param_t(CrossRef<SymR2>);
        else neml_assert(false, "Unsupported parameter type for parameter ", n->fullpath());

        break;
      }
    neml_assert(found, "Unused parameter ", n->fullpath());
  }
}
} // namespace neml2
