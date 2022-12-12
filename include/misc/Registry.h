#pragma once

#include "misc/InputParser.h"

#include <string>
#include <map>

namespace neml2
{
#define combineNames(X, Y) X##Y

/// Add a NEML2Object to the registry.  classname is the (unquoted)
/// c++ class.  Each object/class should only be registered once.
#define register_NEML2_object(classname)                                                           \
  static char combineNames(dummyvar_for_registering_obj_##classname, __COUNTER__) =                \
      Registry::add<classname>(#classname)

class NEML2Object;

using BuildPtr = std::shared_ptr<NEML2Object> (*)(InputParameters & params);

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
    get_registry()._objects[name] = &build<T>;
    return 0;
  }

  /// Return all NEML2Objects in the registry.
  static const std::map<std::string, BuildPtr> & objects() { return get_registry()._objects; }

private:
  Registry(){};

  template <typename T>
  static std::shared_ptr<NEML2Object> build(InputParameters & params)
  {
    return std::make_shared<T>(params);
  }

  std::map<std::string, BuildPtr> _objects;
};
} // namespace neml2
