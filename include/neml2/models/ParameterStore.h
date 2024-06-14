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

#include "neml2/base/NEML2Object.h"
#include "neml2/base/OptionSet.h"
#include "neml2/base/Storage.h"
#include "neml2/tensors/TensorValue.h"

namespace neml2
{
class VariableBase;

/// Interface for object which can store parameters
class ParameterStore
{
public:
  ParameterStore(const OptionSet & options, NEML2Object * object);

  /// @returns the buffer storage
  ///@{
  const Storage<std::string, TensorValueBase> & named_parameters() const
  {
    return const_cast<ParameterStore *>(this)->named_parameters();
  }
  Storage<std::string, TensorValueBase> & named_parameters();
  ///}@

  /// Get a writable reference of a parameter
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  T & get_parameter(const std::string & name);

  /// Whether this parameter store has any nonlinear parameter
  bool has_nl_param() const { return !_nl_params.empty(); }

  /// Get all nonlinear parameters
  const std::map<std::string, const VariableBase *> & nl_params() const { return _nl_params; }

  /**
   * @brief Query the existence of a nonlinear parameter
   *
   * @return const VariableBase* Pointer to the VariableBase if the parameter associated with the
   * given parameter name is nonlinear. Returns nullptr otherwise.
   */
  const VariableBase * nl_param(const std::string &) const;

protected:
  /**
   * @brief Send parameters to options
   *
   * @param options The target options
   */
  virtual void send_parameters_to(const torch::TensorOptions & options);

  /**
   * @brief Declare a parameter.
   *
   * Note that all parameters are stored in the host (the object exposed to users). An object may be
   * used multiple times in the host, and the same parameter may be declared multiple times. That is
   * allowed, but only the first call to declare_parameter constructs the parameter value, and
   * subsequent calls only returns a reference to the existing parameter.
   *
   * @tparam T Buffer type. See @ref statically-shaped-tensor for supported types.
   * @param name Buffer name
   * @param rawval Buffer value
   * @return Reference to buffer
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const T & rawval);

  /**
   * @brief Declare a parameter.
   *
   * Note that all parameters are stored in the host (the object exposed to users). An object may be
   * used multiple times in the host, and the same parameter may be declared multiple times. That is
   * allowed, but only the first call to declare_parameter constructs the parameter value, and
   * subsequent calls only returns a reference to the existing parameter.
   *
   * @tparam T Parameter type. See @ref statically-shaped-tensor for supported types.
   * @param name Name of the model parameter.
   * @param input_option_name Name of the input option that defines the value of the model
   * parameter.
   * @return T The value of the registered model parameter.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const std::string & input_option_name);

private:
  NEML2Object * _object;

  /**
   * @brief Parsed input file options. These options could be convenient when we look up a
   * cross-referenced tensor value by its name.
   *
   */
  const OptionSet _options;

  /// The actual storage for all the parameters
  Storage<std::string, TensorValueBase> _param_values;

  /// @brief Map from nonlinear parameter names to their corresponding variable views
  std::map<std::string, const VariableBase *> _nl_params;
};

template <typename T, typename>
T &
ParameterStore::get_parameter(const std::string & name)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");

  auto base_ptr = _param_values.query_value(name);
  neml_assert(base_ptr, "Parameter named ", name, " does not exist.");
  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert_dbg(ptr, "Internal error: Failed to cast parameter to a concrete type.");
  return ptr->value();
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name, const T & rawval)
{
  if (_object->host() != _object)
    return _object->host<ParameterStore>()->declare_parameter(_object->name() + "." + name, rawval);

  // If the parameter already exists, return its reference
  if (_param_values.has_key(name))
    return get_parameter<T>(name);

  auto val = std::make_unique<TensorValue<T>>(rawval);
  auto base_ptr = _param_values.set_pointer(name, std::move(val));
  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert(ptr, "Internal error: Failed to cast parameter to a concrete type.");
  return ptr->value();
}

} // namespace neml2
