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

#include "neml2/base/OptionSet.h"
#include "neml2/base/UniqueVector.h"
#include "neml2/tensors/BatchTensorValue.h"

namespace neml2
{
/// Interface for object which can store parameters
class ParameterStore
{
public:
  ParameterStore(const OptionSet & options)
    : _options(options)
  {
  }

  bool has_parameter(const std::string & name) const { return _param_ids.count(name); }

  bool has_nonlinear_parameter(const std::string & name) const { return _nl_params.count(name); }

  /**
   * @brief Get the named parameters
   *
   * @return A map from parameter name to parameter value
   */
  virtual std::map<std::string, BatchTensor> named_parameters() const;

  /// Get a parameter's value
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & get_parameter(const std::string & name) const;

protected:
  /**
   * @brief Send buffers to device
   *
   * @param device The target device
   */
  virtual void send_parameters_to(const torch::Device & device);

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

private:
  /**
   * @brief Parsed input file options. These options could be convenient when we look up a
   * cross-referenced tensor value by its name.
   *
   */
  const OptionSet & _options;

  /// Map from parameter name to ID
  std::map<std::string, size_t> _param_ids;

  /// Map from parameter ID to name
  std::vector<std::string> _param_names;

  /// The actual storage for all the parameters
  UniqueVector<BatchTensorValueBase> _param_values;

  /// Accessors for all the *nonlinear* parameters
  std::map<std::string, LabeledAxisAccessor> _nl_params;
};

inline std::map<std::string, BatchTensor>
ParameterStore::named_parameters() const
{
  std::map<std::string, BatchTensor> params;

  for (const auto & n : _param_names)
    params.emplace(n, _param_values[_param_ids.at(n)]);

  return params;
}

template <typename T, typename>
const T &
ParameterStore::get_parameter(const std::string & name) const
{
  auto id = _param_ids.at(name);
  const auto & base_prop = _param_values[id];
  const auto prop = dynamic_cast<const BatchTensorValue<T> *>(&base_prop);
  neml_assert_dbg(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

inline void
ParameterStore::send_parameters_to(const torch::Device & device)
{
  for (auto && [name, id] : _param_ids)
    _param_values[id].to(device);
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name, const T & rawval)
{
  neml_assert(std::find(_param_names.begin(), _param_names.end(), name) == _param_names.end(),
              "Trying to declare a parameter named ",
              name,
              " that already exists.");

  auto val = std::make_unique<BatchTensorValue<T>>(rawval);
  _param_ids.emplace(name, _param_ids.size());
  _param_names.push_back(name);
  auto & base_prop = _param_values.add_pointer(std::move(val));
  auto prop = dynamic_cast<BatchTensorValue<T> *>(&base_prop);
  neml_assert(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

} // namespace neml2
