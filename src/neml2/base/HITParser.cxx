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

namespace neml2
{
void
HITParser::parse(const std::string & filename)
{
  std::ifstream file(filename);
  neml_assert(file.is_open(), "Unable to open file ", filename);

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string input = buffer.str();

  _root.reset(dynamic_cast<hit::Section *>(hit::parse("Hit parser", input)));
}

ParameterCollection
HITParser::parameters() const
{
  ParameterCollection all_params;

  for (const auto & section : Factory::pipeline)
  {
    auto section_node = _root->find(section);
    if (section_node)
    {
      auto objects = section_node->children(hit::NodeType::Section);
      for (auto object : objects)
      {
        // The object name is its node path
        std::string name = object->path();

        // There is a special field reserved for object type
        std::string type = object->param<std::string>("type");

        // Retrieve the expected parameters of this object
        ParameterSet params = Registry::expected_params(type);
        params.set<std::string>("name") = name;
        params.set<std::string>("type") = type;

        // Extract other parameters
        ExtractParamsWalker epw(params);
        object->walk(&epw);

        all_params[section][name] = params;
      }
    }
  }

  return all_params;
}

void
HITParser::ExtractParamsWalker::walk(const std::string & fullpath,
                                     const std::string & nodepath,
                                     hit::Node * n)
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
    for (auto & [name, param] : _params)
      if (name == nodepath)
      {
        found = true;

        if (false)
          ;
        extract_param_t(bool);
        extract_param_t(int);
        extract_param_t(unsigned int);
        extract_param_t(Real);
        extract_param_t(std::string);
        else neml_assert(false, "Unsupported parameter type for parameter ", fullpath);

        break;
      }
    neml_assert(found, "Unused parameter ", fullpath);
  }
}
} // namespace neml2
