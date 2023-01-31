#include "neml2/base/Factory.h"

namespace neml2
{
std::vector<std::string> Factory::pipeline = {"Solvers", "Models"};

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
