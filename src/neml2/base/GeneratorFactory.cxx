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

#include "neml2/base/GeneratorFactory.h"

namespace neml2
{
GeneratorFactory &
GeneratorFactory::get()
{
  static GeneratorFactory factory_singleton;
  return factory_singleton;
}

ParameterCollection
GeneratorFactory::generate(hit::Node * root)
{
  _root.reset(root);
  HITParser parser;

  ParameterCollection all_params;

  for (auto [syntax, builder] : GeneratorRegistry::generators())
  {
    // Some other generator may already requested this generator, and so it may have already been
    // created (and executed).
    if (_generators.count(syntax))
      all_params.merge(_generators[syntax]->generate());
    else
    {
      auto section_node = root->find(syntax);
      if (section_node)
      {
        auto params = parser.extract_generator_parameters(section_node);
        auto generator = builder(params, section_node);
        _generators[syntax] = generator;
        all_params.merge(_generators[syntax]->generate());
      }
    }
  }

  return all_params;
}
} // namespace neml2
