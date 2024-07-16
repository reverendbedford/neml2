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

#include "neml2/models/ParameterStore.h"
#include "neml2/models/NonlinearParameter.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/Variable.h"

namespace neml2
{
ParameterStore::ParameterStore(const OptionSet & options, NEML2Object * object)
  : _object(object),
    _options(options)
{
}

void
ParameterStore::send_parameters_to(const torch::TensorOptions & options)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");

  for (auto && [name, param] : _param_values)
    param.to_(options);
}

void
ParameterStore::set_parameter(const std::string & name, const Tensor & value)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");

  neml_assert(named_parameters().has_key(name), "There is no parameter named ", name);
  named_parameters()[name] = value;
}

void
ParameterStore::set_parameters(const std::map<std::string, Tensor> & param_values)
{
  for (auto & [name, value] : param_values)
    set_parameter(name, value);
}

TensorValueBase &
ParameterStore::get_parameter(const std::string & name)
{
  auto base_ptr = _param_values.query_value(name);
  neml_assert(base_ptr, "Parameter named ", name, " does not exist.");
  return *base_ptr;
}

const VariableBase *
ParameterStore::nl_param(const std::string & name) const
{
  return _nl_params.count(name) ? _nl_params.at(name) : nullptr;
}

Storage<std::string, TensorValueBase> &
ParameterStore::named_parameters()
{
  neml_assert(_object->host() == _object,
              "named_parameters() should only be called on the host model.");
  return _param_values;
}

std::map<std::string, const VariableBase *>
ParameterStore::named_nonlinear_parameters(bool recursive) const
{
  if (!recursive)
    return _nl_params;

  auto model = dynamic_cast<const Model *>(this);
  neml_assert(model, "Only models support recursive nonlinear parameter declaration");

  auto all_nl_params = _nl_params;
  for (auto submodel : model->registered_models())
  {
    auto sub_nl_params = submodel->named_nonlinear_parameters(true);
    all_nl_params.insert(sub_nl_params.begin(), sub_nl_params.end());
  }
  for (auto && [pname, pmodel] : _nl_param_models)
  {
    auto sub_nl_params = pmodel->named_nonlinear_parameters(true);
    all_nl_params.insert(sub_nl_params.begin(), sub_nl_params.end());
  }
  return all_nl_params;
}

std::map<std::string, Model *>
ParameterStore::named_nonlinear_parameter_models(bool recursive) const
{
  if (!recursive)
    return _nl_param_models;

  auto model = dynamic_cast<const Model *>(this);
  neml_assert(model, "Only models support recursive nonlinear parameter declaration");

  auto all_nl_param_models = _nl_param_models;
  for (auto submodel : model->registered_models())
  {
    auto sub_nl_param_models = submodel->named_nonlinear_parameter_models(true);
    all_nl_param_models.insert(sub_nl_param_models.begin(), sub_nl_param_models.end());
  }
  for (auto && [pname, pmodel] : _nl_param_models)
  {
    auto sub_nl_param_models = pmodel->named_nonlinear_parameter_models(true);
    all_nl_param_models.insert(sub_nl_param_models.begin(), sub_nl_param_models.end());
  }
  return all_nl_param_models;
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name, const std::string & input_option_name)
{
  if (_options.contains<T>(input_option_name))
    return declare_parameter(name, _options.get<T>(input_option_name));
  else if (_options.contains<CrossRef<T>>(input_option_name))
  {
    try
    {
      return declare_parameter(name, T(_options.get<CrossRef<T>>(input_option_name)));
    }
    catch (const NEMLException & e1)
    {
      try
      {
        // Handle the case of *nonlinear* parameter.
        // Note that nonlinear parameter should only exist inside a Model.
        auto model = dynamic_cast<Model *>(this);
        neml_assert(model,
                    "Trying to declare a parameter named ",
                    name,
                    ". It is not a plain tensor value nor a cross-referenced parameter "
                    "value. Hence I am guessing you are declaring a *nonlinear* parameter. "
                    "However, nonlinear parameter should only be declared by a model, and this "
                    "object does not appear to be a model.");

        OptionSet extra_opts;
        extra_opts.set<NEML2Object *>("_host") = model->host();
        extra_opts.set<bool>("_enable_AD") = model->input_options().get<bool>("_enable_AD");
        auto pname = _options.get<CrossRef<T>>(input_option_name).raw();
        auto & nl_param = Factory::get_object<NonlinearParameter<T>>(
            "Models", pname, extra_opts, /*force_create=*/false);
        model->template declare_input_variable<T>(VariableName(pname));
        _nl_params[name] = &nl_param.param();
        _nl_param_models[name] = &nl_param;
        return nl_param.param().value();
      }
      catch (const NEMLException & e2)
      {
        std::cerr << e1.what() << std::endl;
        std::cerr << e2.what() << std::endl;
      }
    }
  }

  throw NEMLException(
      "Trying to register parameter named " + name + " from input option named " +
      input_option_name + " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct parameter name, option name, and parameter type. Note "
      "that the parameter type can either be a plain type, a cross-reference, or a nonlinear "
      "parameter.");
}

#define PARAMETERSTORE_INTANTIATE_PrimitiveTensor(T)                                               \
  template const T & ParameterStore::declare_parameter<T>(const std::string &, const std::string &)
FOR_ALL_PRIMITIVETENSOR(PARAMETERSTORE_INTANTIATE_PrimitiveTensor);
} // namespace neml2
