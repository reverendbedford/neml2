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

#include <torch/torch.h>
#include "neml2/base/UniqueVector.h"
#include "neml2/models/ParameterValue.h"

namespace neml2
{

/// Interface for object which can store buffers
template <typename Base>
class ContainsParameters : public Base
{
public:
  using Base::Base;
  /**
   * @brief Send buffers to device
   *
   * @param device The target device
   */
  virtual void to(const torch::Device & device) override;

  bool has_parameter(const std::string & name) const { return _param_ids.count(name); }

  bool has_nonlinear_parameter(const std::string & name) const { return _nl_params.count(name); }

  /**
   * @brief (Recursively) get the named model parameters
   *
   * If \p recurse is set true, then each sub-model's parameters are prepended by the model name
   * followed by a dot ".". This is consistent with torch::nn::Module's naming convention.
   *
   * @param recurse Whether to recursively retrieve parameter names of sub-models.
   * @return A map from parameter name to parameter value
   */
  virtual std::map<std::string, BatchTensor> named_parameters(bool recurse = false) const override;

  /// Get a parameter's value
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & get_parameter(const std::string & name) const;

protected:
  /// Get the accessor for a given nonlinear parameter
  const LabeledAxisAccessor & nl_param(const std::string & name) const
  {
    return _nl_params.at(name);
  }

  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const T & rawval);

  /**
   * @brief Declare a model parameter.
   *
   * @tparam T Parameter type. See @ref primitive for supported types.
   * @param name Name of the model parameter.
   * @param input_option_name Name of the input option that defines the value of the model
   * parameter.
   * @return T The value of the registered model parameter.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const std::string & input_option_name);

  std::map<std::string, size_t> _param_ids;
  std::vector<std::string> _param_names;
  UniqueVector<ParameterValueBase> _param_values;

  std::map<std::string, LabeledAxisAccessor> _nl_params;
};

template <typename Base>
template <typename T, typename>
const T &
ContainsParameters<Base>::get_parameter(const std::string & name) const
{
  auto id = _param_ids.at(name);
  const auto & base_prop = _param_values[id];
  const auto prop = dynamic_cast<const ParameterValue<T> *>(&base_prop);
  neml_assert_dbg(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

template <typename Base>
template <typename T, typename>
const T &
ContainsParameters<Base>::declare_parameter(const std::string & name, const T & rawval)
{
  neml_assert(std::find(_param_names.begin(), _param_names.end(), name) == _param_names.end(),
              "Trying to declare a parameter named ",
              name,
              " that already exists.");

  auto val = std::make_unique<ParameterValue<T>>(rawval);
  _param_ids.emplace(name, _param_ids.size());
  _param_names.push_back(name);
  auto & base_prop = _param_values.add_pointer(std::move(val));
  auto prop = dynamic_cast<ParameterValue<T> *>(&base_prop);
  neml_assert(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

} // namespace neml2