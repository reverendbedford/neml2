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
 * The factory is responsible for:
 * 1. retriving a `NEML2Object` given the object name as a `std::string`
 * 2. creating a `NEML2Object` given the type of the `NEML2Object` as a `std::string`.
 */
class Factory
{
public:
  /// The sequence which we use to manufacture objects.
  static std::vector<std::string> pipeline;

  /// Get the global `Factory` singleton.
  static Factory & get();

  /**
   * @brief Retrive an object pointer under the given section with the given object name.
   *
   * An exception is thrown if
   * - the object with the given name does not exist, or
   * - the object with the given name exists but does not have the correct type (e.g., dynamic case
   * fails).
   *
   * @tparam T The type of the `NEML2Object`
   * @param section The section name under which the search happens.
   * @param name The name of the object to retrieve.
   * @return std::shared_ptr<T> The object pointer.
   */
  template <class T>
  static std::shared_ptr<T> get_object_ptr(const std::string & section, const std::string & name);

  /**
   * @brief Retrive an object reference under the given section with the given object name.
   *
   * An exception is thrown if
   * - the object with the given name does not exist, or
   * - the object with the given name exists but does not have the correct type (e.g., dynamic case
   * fails).
   *
   * @tparam T The type of the `NEML2Object`
   * @param section The section name under which the search happens.
   * @param name The name of the object to retrieve.
   * @return T & The object reference.
   */
  template <class T>
  static T & get_object(const std::string & section, const std::string & name);

  /**
   * @brief Manufacture all recognized objects in a parameter collection.
   *
   * @param all_params The collection of all the parameters of the objects to be manufactured.
   */
  void manufacture(const ParameterCollection & all_params);

  /**
   * @brief Manufacture a single `NEML2Object`.
   *
   * @param section The section which the object to be manufactured belongs to.
   * @param params The parameters of the object.
   */
  void create_object(const std::string & section, const ParameterSet & params);

  /**
   * @brief List all the manufactured objects.
   *
   * @param os The stream to write to.
   */
  void print(std::ostream & os = std::cout) const;

  /**
   * @brief Destruct all the objects.
   *
   */
  void clear() { _objects.clear(); }

private:
  /// The collection of all the parameters of the objects to be manufactured.
  ParameterCollection _all_params;

  // Manufactured objects. The key of the outer map is the section name, and the key of the inner
  // map is the object name.
  std::map<std::string, std::map<std::string, std::shared_ptr<NEML2Object>>> _objects;
};

template <class T>
inline std::shared_ptr<T>
Factory::get_object_ptr(const std::string & section, const std::string & name)
{
  auto & factory = Factory::get();

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

  return Factory::get_object_ptr<T>(section, name);
}

template <class T>
inline T &
Factory::get_object(const std::string & section, const std::string & name)
{
  return *Factory::get_object_ptr<T>(section, name);
}
} // namespace neml2
