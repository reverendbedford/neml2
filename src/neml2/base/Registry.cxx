#include "neml2/base/Registry.h"
#include "neml2/models/Model.h"

namespace neml2
{
Registry &
Registry::get_registry()
{
  static Registry registry_singleton;
  return registry_singleton;
}

void
Registry::add_inner(const std::string & name, const ParameterSet & params, BuildPtr build_ptr)
{
  get_registry()._expected_params[name] = params;
  get_registry()._objects[name] = build_ptr;
}

void
Registry::print(std::ostream & os) const
{
  for (auto & object : _objects)
    os << object.first << std::endl;
}
} // namespace neml2
