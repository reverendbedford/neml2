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
#include "neml2/base/Factory.h"

namespace neml2
{
std::vector<std::string> Factory::pipeline = {
    "Tensors", "Solvers", "Predictors", "Models", "Drivers"};

Factory &
Factory::get_factory()
{
  static Factory factory_singleton;
  return factory_singleton;
}

void
Factory::manufacture(const ParameterCollection & all_params)
{
  _all_params = all_params;

  for (const auto & section : Factory::pipeline)
    for (const auto & params : _all_params[section])
      create_object(section, params.second);
}

void
Factory::create_object(const std::string & section, const ParameterSet & params)
{
  const std::string & name = params.get<std::string>("name");
  const std::string & type = params.get<std::string>("type");

  // Some other object might have already requested the existence of this object, at which time we
  // have already created it. So don't bother doing anything again.
  if (_objects[section].count(name))
    return;

  auto builder = Registry::builder(type);
  _objects[section].emplace(name, (*builder)(params));
}

void
Factory::print(std::ostream & os) const
{
  for (auto & [section, objects] : _objects)
  {
    os << section << ":" << std::endl;
    for (auto & object : objects)
      os << " " << object.first << std::endl;
  }
}
} // namespace neml2
