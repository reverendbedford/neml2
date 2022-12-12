#include "misc/Registry.h"
#include "misc/NEML2Object.h"

namespace neml2
{
Registry &
Registry::get_registry()
{
  static Registry registry_singleton;
  return registry_singleton;
}
} // namespace neml2
