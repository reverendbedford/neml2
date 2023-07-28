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
#pragma once

#include "neml2/base/ParameterSet.h"
#include "hit.h"

namespace neml2
{
/// Add a Generator to the registry and associate it with a syntax.  classname is the
/// (unquoted) c++ class.  Each object/class should only be registered once.
#define register_NEML2_generator(syntax, classname)                                                \
  static char dummyvar_for_registering_generator_##classname =                                     \
      GeneratorRegistry::add<classname>(syntax)

class Generator;

using GeneratorBuildPtr = std::shared_ptr<Generator> (*)(const ParameterSet &, hit::Node *);

/**
The generator registry is used as a global singleton to collect information on all available Genetor
for use during input file parsing
*/
class GeneratorRegistry
{
public:
  /// Get the global GeneratorRegistry singleton.
  static GeneratorRegistry & get();

  /// Add information on a Generator to the registry.
  template <typename T>
  static char add(std::string name)
  {
    add_inner(name, T::expected_params(), &build<T>);
    return 0;
  }

  /// Return the registered generators
  static const std::map<std::string, GeneratorBuildPtr> & generators();

  /// Return the build method pointer of a specific registered generator
  static GeneratorBuildPtr builder(const std::string & name);

  /// Return the expected parameters of a specific registered generator
  static ParameterSet expected_params(const std::string & name);

  /// List all registered objects
  static void print(std::ostream & os = std::cout);

private:
  GeneratorRegistry() {}

  static void add_inner(const std::string &, const ParameterSet &, GeneratorBuildPtr);

  template <typename T>
  static std::shared_ptr<Generator> build(const ParameterSet & params, hit::Node * root)
  {
    return std::make_shared<T>(params, root);
  }

  std::map<std::string, GeneratorBuildPtr> _generators;
  std::map<std::string, ParameterSet> _expected_params;
};
} // namespace neml2
