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
class ContainsBuffers
{
public:
  /**
   * @brief Send buffers to device
   *
   * @param device The target device
   */
  virtual void to(const torch::Device & device);

  /// Get a buffer's value
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & get_buffer(const std::string & name) const;

protected:
  std::map<std::string, size_t> _buffer_ids;
  std::vector<std::string> _buffer_names;
  UniqueVector<ParameterValueBase> _buffer_values;

  /// @brief  Declare a buffer
  /// @tparam T type of buffer
  /// @param name string name
  /// @param rawval buffer value
  /// @return reference to buffer
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const T & rawval);

  /**
   * @brief Declare a model buffer.
   *
   * @tparam T Buffer type. See @ref primitive for supported types.
   * @param name Name of the model buffer.
   * @param input_option_name Name of the input option that defines the value of the model
   * buffer.
   * @return T The value of the registered model buffer.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const std::string & input_option_name);
};

template <typename Base>
template <typename T, typename>
const T &
ContainsBuffers<Base>::get_buffer(const std::string & name) const
{
  auto id = _buffer_ids.at(name);
  const auto & base_prop = _buffer_values[id];
  const auto prop = dynamic_cast<const ParameterValue<T> *>(&base_prop);
  neml_assert_dbg(prop, "Internal error, buffer cast failure.");
  return prop->get();
}

template <typename Base>
template <typename T, typename>
const T &
ContainsBuffers<Base>::declare_buffer(const std::string & name, const T & rawval)
{
  neml_assert(std::find(_buffer_names.begin(), _buffer_names.end(), name) == _buffer_names.end(),
              "Trying to declare a buffer named ",
              name,
              " that already exists.");

  auto val = std::make_unique<ParameterValue<T>>(rawval);
  _buffer_ids.emplace(name, _buffer_ids.size());
  _buffer_names.push_back(name);
  auto & base_prop = _buffer_values.add_pointer(std::move(val));
  auto prop = dynamic_cast<ParameterValue<T> *>(&base_prop);
  neml_assert(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

template <typename Base>
template <typename T, typename>
const T &
ContainsBuffers<Base>::declare_buffer(const std::string & name,
                                      const std::string & input_option_name)
{
  if (Base::options().template contains<T>(input_option_name))
    return declare_buffer(name, Base::options().template get<T>(input_option_name));
  else if (Base::options().template contains<CrossRef<T>>(input_option_name))
    return declare_buffer(name, T(Base::options().template get<CrossRef<T>>(input_option_name)));

  throw NEMLException(
      "Trying to register buffer named " + name + " from input option named " + input_option_name +
      " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct buffer name, option name, and buffer type. Note that "
      "the buffer type can either be a plain type, a cross-reference, or an interpolator.");
}

} // namespace neml2