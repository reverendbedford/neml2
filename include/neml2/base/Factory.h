#pragma once

#include "neml2/base/Registry.h"
#include "neml2/base/NEML2Object.h"

namespace neml2
{
using ParameterCollection = std::map<std::string, std::map<std::string, ParameterSet>>;

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
  if (factory._objects.at(section).count(name))
    return std::dynamic_pointer_cast<T>(factory._objects[section][name]);

  // Otherwise try to create it
  for (const auto & params : factory._all_params[section])
    if (params.first == name)
    {
      factory.create_object(section, params.second);
      break;
    }

  neml_assert(factory._objects.at(section).count(name),
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
