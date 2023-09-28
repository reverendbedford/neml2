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
#include "neml2/base/Factory.h"
#include "neml2/base/CrossRef.h"
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/tensors/tensors.h"
#include <memory>

namespace neml2
{
OptionCollection
HITParser::parse(const std::string & filename, const std::string & additional_input) const
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

  // Explode the tree
  hit::explode(root);

  // Handle additional input (they could be coming from cli args)
  auto cli_root = hit::parse("Hit cliargs", additional_input);
  hit::explode(cli_root);
  hit::merge(cli_root, root);

  // Preevaluate the input
  hit::BraceExpander expander;
  hit::EnvEvaler env;
  hit::RawEvaler raw;
  expander.registerEvaler("env", env);
  expander.registerEvaler("raw", raw);
  root->walk(&expander);

  // Loop over each known section and extract options for each object
  OptionCollection all_options;
  for (const auto & section : Factory::pipeline)
  {
    auto section_node = root->find(section);
    if (section_node)
    {
      auto objects = section_node->children(hit::NodeType::Section);
      for (auto object : objects)
      {
        auto options = extract_object_options(object);
        all_options[section][options.get<std::string>("name")] = options;
      }
    }
  }

  return all_options;
}

OptionSet
HITParser::extract_object_options(hit::Node * object) const
{
  // The object name is its node path
  std::string name = object->path();
  // There is a special field reserved for object type
  std::string type = object->param<std::string>("type");
  // Extract the options
  auto options = Registry::expected_options(type);
  extract_options(object, options);
  // Also fill in the special fields, e.g., name, type
  options.set<std::string>("name") = name;
  options.set<std::string>("type") = type;

  return options;
}

void
HITParser::extract_options(hit::Node * object, OptionSet & options) const
{
  for (auto node : object->children(hit::NodeType::Field))
    extract_option(node, options);
}

void
HITParser::extract_option(hit::Node * n, OptionSet & options) const
{
#define extract_option_base(ptype, method)                                                         \
  else if (option->type() ==                                                                       \
           utils::demangle(                                                                        \
               typeid(ptype).name())) dynamic_cast<OptionSet::Option<ptype> *>(option.get())       \
      ->set() = method(n->strVal())

#define extract_option_t(ptype)                                                                    \
  extract_option_base(ptype, utils::parse<ptype>);                                                 \
  extract_option_base(std::vector<ptype>, utils::parse_vector<ptype>);                             \
  extract_option_base(std::vector<std::vector<ptype>>, utils::parse_vector_vector<ptype>)

  if (n->type() == hit::NodeType::Field)
  {
    bool found = false;
    for (auto & [name, option] : options)
      if (name == n->path())
      {
        found = true;

        if (false)
          ;
        extract_option_t(TorchShape);
        extract_option_t(bool);
        extract_option_t(int);
        extract_option_t(unsigned int);
        extract_option_t(TorchSize);
        extract_option_t(Real);
        extract_option_t(std::string);
        extract_option_t(LabeledAxisAccessor);
        extract_option_t(CrossRef<torch::Tensor>);
        extract_option_t(CrossRef<BatchTensor>);
        extract_option_t(CrossRef<Scalar>);
        extract_option_t(CrossRef<Vec>);
        extract_option_t(CrossRef<Rot>);
        extract_option_t(CrossRef<R2>);
        extract_option_t(CrossRef<SR2>);
        extract_option_t(CrossRef<R3>);
        extract_option_t(CrossRef<SFR3>);
        extract_option_t(CrossRef<R4>);
        extract_option_t(CrossRef<SSR4>);
        extract_option_t(CrossRef<R5>);
        extract_option_t(CrossRef<SSFR5>);
        // LCOV_EXCL_START
        else neml_assert(false, "Unsupported option type for option ", n->fullpath());
        // LCOV_EXCL_STOP

        break;
      }
    neml_assert(found, "Unused option ", n->fullpath());
  }
}
} // namespace neml2
