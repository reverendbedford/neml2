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
/// Interface for object which can store buffers
class BufferStore
{
public:
  BufferStore(const OptionSet & options)
    : _options(options)
  {
  }

  /**
   * @brief Get the named buffers
   *
   * @return A map from buffer name to buffer value
   */
  virtual std::map<std::string, BatchTensor> named_buffers() const;

  /// Get a buffer's value
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & get_buffer(const std::string & name) const;

protected:
  /**
   * @brief Send buffers to device
   *
   * @param device The target device
   */
  virtual void send_buffers_to(const torch::Device & device);

  /**
   * @brief Declare a model buffer.
   *
   * @tparam T Buffer type. See @ref primitive for supported types.
   * @param name Buffer name
   * @param rawval Buffer value
   * @return Reference to buffer
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const T & rawval);

  /**
   * @brief Declare a model buffer.
   *
   * @tparam T Buffer type. See @ref primitive for supported types.
   * @param name Buffer name
   * @param input_option_name Name of the input option that defines the value of the model
   * buffer.
   * @return T Reference to buffer
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const std::string & input_option_name);

private:
  /**
   * @brief Parsed input file options. These options could be convenient when we look up a
   * cross-referenced tensor value by its name.
   *
   */
  const OptionSet & _options;

  /// Map from buffer name to ID
  std::map<std::string, size_t> _buffer_ids;

  /// Map from buffer ID to name
  std::vector<std::string> _buffer_names;

  /// The actual storage for all the buffers
  UniqueVector<BatchTensorValueBase> _buffer_values;
};

inline std::map<std::string, BatchTensor>
BufferStore::named_buffers() const
{
  std::map<std::string, BatchTensor> buffers;

  for (const auto & n : _buffer_names)
    buffers.emplace(n, _buffer_values[_buffer_ids.at(n)]);

  return buffers;
}

template <typename T, typename>
const T &
BufferStore::get_buffer(const std::string & name) const
{
  auto id = _buffer_ids.at(name);
  const auto & base_prop = _buffer_values[id];
  const auto prop = dynamic_cast<const BatchTensorValue<T> *>(&base_prop);
  neml_assert_dbg(prop, "Internal error, buffer cast failure.");
  return prop->get();
}

inline void
BufferStore::send_buffers_to(const torch::Device & device)
{
  for (auto && [name, id] : _buffer_ids)
    _buffer_values[id].to(device);
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const T & rawval)
{
  neml_assert(std::find(_buffer_names.begin(), _buffer_names.end(), name) == _buffer_names.end(),
              "Trying to declare a buffer named ",
              name,
              " that already exists.");

  auto val = std::make_unique<BatchTensorValue<T>>(rawval);
  _buffer_ids.emplace(name, _buffer_ids.size());
  _buffer_names.push_back(name);
  auto & base_prop = _buffer_values.add_pointer(std::move(val));
  auto prop = dynamic_cast<BatchTensorValue<T> *>(&base_prop);
  neml_assert(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const std::string & input_option_name)
{
  if (_options.contains<T>(input_option_name))
    return declare_buffer(name, _options.get<T>(input_option_name));
  else if (_options.contains<CrossRef<T>>(input_option_name))
    return declare_buffer(name, T(_options.get<CrossRef<T>>(input_option_name)));

  throw NEMLException(
      "Trying to register buffer named " + name + " from input option named " + input_option_name +
      " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct buffer name, option name, and buffer type. Note that "
      "the buffer type can either be a plain type, a cross-reference, or an interpolator.");
}

} // namespace neml2
