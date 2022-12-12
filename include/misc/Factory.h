#pragma once

#include "misc/Registry.h"
#include "models/Model.h"

namespace neml2
{
/**
The factory is responsible for:
1. retriving a Model given the object name as a `std::string`
2. creating a Model given the type of the Model as a `std::string`
*/
class Factory
{
public:
  /// Get the global Registry singleton.
  static Factory & get_factory();

  /// Manufacture all recognized objects in a param tree
  void manufacture(hit::Node * all_params);

  /// Retrive a `Model` with the given object name
  std::shared_ptr<Model> get(const std::string & name);

  /// Retrive a Model of type T under the [Models] section
  template <class T>
  static const T & get_model(const std::string & name)
  {
    auto model = Factory::get_factory().get(name);
    return *dynamic_cast<T *>(model.get());
  }

private:
  /// Manufacture a model
  void create_model(hit::Node * model_param);

  hit::Node * _all_params;

  std::map<std::string, std::shared_ptr<Model>> _models;
};
} // namespace neml2
