#include "misc/Factory.h"

namespace neml2
{
Factory &
Factory::get_factory()
{
  static Factory factory_singleton;
  return factory_singleton;
}

void
Factory::manufacture(hit::Node * all_params)
{
  _all_params = all_params;

  // Add Models
  auto model_params = _all_params->find("Models")->children(hit::NodeType::Section);
  for (auto model_param : model_params)
    create_model(model_param);

  // Other recognized sections go here...
}

std::shared_ptr<Model>
Factory::get(const std::string & name)
{
  // Easy if it already exists
  if (_models.count(name))
    return _models[name];

  // Otherwise try to create it
  auto model_params = _all_params->find("Models")->find(name)->children(hit::NodeType::Section);
  neml_assert(model_params.size() == 1, "Found duplicate models with the same name");
  create_model(model_params[0]);
  return get(name);
}

void
Factory::create_model(hit::Node * model_param)
{
  std::string name = model_param->path();
  std::string type = model_param->param<std::string>("type");
  auto builder = Registry::builder(type);
  _models.emplace(name, (*builder)(*dynamic_cast<InputParameters *>(model_param)));
}
} // namespace neml2
