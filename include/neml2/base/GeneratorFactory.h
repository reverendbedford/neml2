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

#include "neml2/base/GeneratorRegistry.h"
#include "neml2/generators/Generator.h"
#include "neml2/misc/error.h"
#include "neml2/base/ParameterCollection.h"
#include "neml2/base/HITParser.h"

namespace neml2
{
/**
The generator factory is responsible for:
1. retriving a Generator given the object name as a `std::string`
2. creating and running all Generators that can be recognized in an input file
*/
class GeneratorFactory
{
public:
  /// Get the global Registry singleton.
  static GeneratorFactory & get();

  /// Retrive a generator given its syntax
  template <class T>
  static std::shared_ptr<T> get_generator_ptr(const std::string & syntax);

  /// Retrive a generator given its syntax
  template <class T>
  static T & get_generator(const std::string & syntax);

  // Run all recognized generators in an input file. Their generated ParameterCollections are meged
  // into one single ParameterCollection
  ParameterCollection generate(hit::Node * root);

  /// Destruct all the objects
  void clear() { _generators.clear(); }

private:
  std::unique_ptr<hit::Node> _root;

  std::map<std::string, std::shared_ptr<Generator>> _generators;
};

template <class T>
inline std::shared_ptr<T>
GeneratorFactory::get_generator_ptr(const std::string & syntax)
{
  auto & factory = GeneratorFactory::get();

  // Easy if it already exists
  if (factory._generators.count(syntax))
  {
    auto gen = std::dynamic_pointer_cast<T>(factory._generators[syntax]);
    neml_assert(gen != nullptr,
                "Found generator with syntax ",
                syntax,
                ". But dynamic cast failed. Did you specify the correct generator type?");
    return gen;
  }

  // Otherwise try to create it
  auto section_node = factory._root->find(syntax);
  if (section_node)
  {
    HITParser parser;
    auto params = parser.extract_generator_parameters(section_node);
    auto generator = GeneratorRegistry::builder(syntax)(params, section_node);
    factory._generators[syntax] = generator;
    return GeneratorFactory::get_generator_ptr<T>(syntax);
  }

  return nullptr;
}

template <class T>
inline T &
GeneratorFactory::get_generator(const std::string & syntax)
{
  return *GeneratorFactory::get_generator_ptr<T>(syntax);
}
} // namespace neml2
