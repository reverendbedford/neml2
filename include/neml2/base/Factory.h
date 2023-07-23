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

#include "neml2/base/Registry.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/misc/error.h"
#include "neml2/base/ParameterCollection.h"

namespace neml2
{
/**
The factory is responsible for:
1. retriving a NEML2Object given the object name as a `std::string`
2. creating a NEML2Object given the type of the NEML2Object as a `std::string`
*/
class Factory
{
public:
  /// The sequence which we use to manufacture objects
  static std::vector<std::string> pipeline;

  /// Get the global Registry singleton.
  static Factory & get_factory();

  /// Retrive an object in the given section with the given object name
  template <class T>
  static std::shared_ptr<T> get_object_ptr(const std::string & section, const std::string & name);

  /// Retrive an object in the given section with the given object name
  template <class T>
  static T & get_object(const std::string & section, const std::string & name);

  /// Manufacture all recognized objects in a param tree
  void manufacture(const ParameterCollection & all_params);

  /// Manufacture a NEML2Object
  void create_object(const std::string & section, const ParameterSet & params);

  /// List all the manufactured objects
  void print(std::ostream & os = std::cout) const;

  /// Destruct all the objects
  void clear() { _objects.clear(); }

private:
  ParameterCollection _all_params;

  std::map<std::string, std::map<std::string, std::shared_ptr<NEML2Object>>> _objects;
};

template <class T>
inline std::shared_ptr<T>
Factory::get_object_ptr(const std::string & section, const std::string & name)
{
  auto & factory = Factory::get_factory();

  // Easy if it already exists
  if (factory._objects.count(section) && factory._objects.at(section).count(name))
  {
    auto obj = std::dynamic_pointer_cast<T>(factory._objects[section][name]);
    neml_assert(obj != nullptr,
                "Found object named ",
                name,
                " under section ",
                section,
                ". But dynamic cast failed. Did you specify the correct object type?");
    return obj;
  }

  // Otherwise try to create it
  for (const auto & params : factory._all_params[section])
    if (params.first == name)
    {
      factory.create_object(section, params.second);
      break;
    }

  neml_assert(factory._objects.count(section) && factory._objects.at(section).count(name),
              "Failed to get object named ",
              name,
              " under section ",
              section);

  return factory.get_object_ptr<T>(section, name);
}

template <class T>
inline T &
Factory::get_object(const std::string & section, const std::string & name)
{
  return *Factory::get_factory().get_object_ptr<T>(section, name);
}
} // namespace neml2
