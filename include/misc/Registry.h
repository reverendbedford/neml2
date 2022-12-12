#pragma once

#include "misc/InputParser.h"

#include <string>
#include <map>

namespace neml2
{
#define combineNames(X, Y) X##Y

/// Add a Model to the registry.  classname is the (unquoted)
/// c++ class.  Each object/class should only be registered once.
#define register_NEML2_object(classname)                                                           \
  static char combineNames(dummyvar_for_registering_obj_##classname, __COUNTER__) =                \
      Registry::add<classname>(#classname)

class Model;

using BuildPtr = std::shared_ptr<Model> (*)(InputParameters & params);

/**
The registry is used as a global singleton to collect information on all available Model for
use in a simulation.
*/
class Registry
{
public:
  /// Get the global Registry singleton.
  static Registry & get_registry();

  /// Add information on a Model to the registry.
  template <typename T>
  static char add(std::string name)
  {
    get_registry()._objects[name] = &build<T>;
    return 0;
  }

  /// Return all registered objects in the registry.
  static const std::map<std::string, BuildPtr> & objects() { return get_registry()._objects; }

  /// Return the build method pointer of a specific registered class
  static BuildPtr builder(const std::string & name)
  {
    neml_assert(
        objects().count(name) > 0,
        name,
        " is not a registered object. Did you forget to register it with register_NEML2_object?");
    return objects().at(name);
  }

private:
  Registry(){};

  template <typename T>
  static std::shared_ptr<Model> build(InputParameters & params)
  {
    return std::make_shared<T>(params);
  }

  std::map<std::string, BuildPtr> _objects;
};
} // namespace neml2
