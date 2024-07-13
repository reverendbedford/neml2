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
#include "neml2/base/Registry.h"
#include "neml2/base/Settings.h"
#include "neml2/models/Model.h"

namespace neml2
{
Model &
get_model(const std::string & mname, bool inference_mode, bool force_create)
{
  OptionSet extra_opts;
  extra_opts.set<bool>("_inference_mode") = inference_mode;
  return Factory::get_object<Model>("Models", mname, extra_opts, force_create);
}

Factory &
Factory::get()
{
  static Factory factory_singleton;
  return factory_singleton;
}

void
Factory::load(const OptionCollection & all_options)
{
  auto & factory = get();

  factory._all_options = all_options;

  // Also apply global settings
  factory._settings = Settings(all_options.settings());
}

void
Factory::create_object(const std::string & section, const OptionSet & options)
{
  const std::string & name = options.name();
  const std::string & type = options.type();

  auto builder = Registry::builder(type);
  auto object = (*builder)(options);
  _objects[section][name].push_back(object);
  object->setup();
}

// LCOV_EXCL_START
void
Factory::print(std::ostream & os)
{
  const auto & all_objects = get()._objects;
  for (auto & [section, objects] : all_objects)
  {
    os << "- " << section << ":" << std::endl;
    for (auto & object : objects)
      os << "   " << object.first << ": " << object.second.size() << std::endl;
  }
}
// LCOV_EXCL_STOP

void
Factory::clear()
{
  get()._objects.clear();
}
} // namespace neml2
