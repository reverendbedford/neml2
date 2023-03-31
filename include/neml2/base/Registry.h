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

#include <string>
#include <map>

namespace neml2
{
/// Add a NEML2Object to the registry.  classname is the (unquoted)
/// c++ class.  Each object/class should only be registered once.
#define register_NEML2_object(classname)                                                           \
  static char dummyvar_for_registering_obj_##classname = Registry::add<classname>(#classname)

/// Add a NEML2Object to the registry and associate it with a given name.  classname is the
/// (unquoted) c++ class.  Each object/class should only be registered once.
#define register_NEML2_object_alt(classname, registryname)                                         \
  static char dummyvar_for_registering_obj_##classname = Registry::add<classname>(registryname)

class NEML2Object;

using BuildPtr = std::shared_ptr<NEML2Object> (*)(const ParameterSet & params);

/**
The registry is used as a global singleton to collect information on all available NEML2Object for
use in a simulation.
*/
class Registry
{
public:
  /// Get the global Registry singleton.
  static Registry & get_registry();

  /// Add information on a NEML2Object to the registry.
  template <typename T>
  static char add(std::string name)
  {
    add_inner(name, T::expected_params(), &build<T>);
    return 0;
  }

  /// Return the expected parameters of a specific registered class
  static ParameterSet expected_params(const std::string & name)
  {
    neml_assert(
        get_registry()._expected_params.count(name) > 0,
        name,
        " is not a registered object. Did you forget to register it with register_NEML2_object?");
    return get_registry()._expected_params.at(name);
  }

  /// Return the build method pointer of a specific registered class
  static BuildPtr builder(const std::string & name)
  {
    neml_assert(
        get_registry()._objects.count(name) > 0,
        name,
        " is not a registered object. Did you forget to register it with register_NEML2_object?");
    return get_registry()._objects.at(name);
  }

  /// List all registered objects
  void print(std::ostream & os = std::cout) const;

private:
  Registry(){};

  static void add_inner(const std::string &, const ParameterSet &, BuildPtr);

  template <typename T>
  static std::shared_ptr<NEML2Object> build(const ParameterSet & params)
  {
    return std::make_shared<T>(params);
  }

  std::map<std::string, ParameterSet> _expected_params;

  std::map<std::string, BuildPtr> _objects;
};
} // namespace neml2
